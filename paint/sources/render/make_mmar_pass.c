#include "../global.h"

// Run one cooked .mmar pass into a texture.
//
// Material Maker renders buffer nodes -- blur, normal_map2, smooth_curvature,
// hbao and the rest -- in separate passes and samples the results. An archive
// carrying such a material is therefore a short list of programs, not one, and
// a host has to be able to evaluate each into a target before the final kernel
// can run.
//
// Everything here already existed: util_render_make_node_preview does the same
// thing for node previews. It reaches the runtime compiler the long way, by
// building a node_shader_t and asking for its source back; a pass is already a
// complete kong program, so it goes straight to gpu_create_shaders_from_kong.
//
// The quad is our own rather than util_render's. That one is in the mesh vertex
// format (short4norm pos, nor, tex, col) because the preview path reuses the
// mesh pipeline, while a pass declares `struct vert_in { pos: float2; }` -- an
// archive should not have to encode one host's mesh layout to be renderable.

static gpu_buffer_t *mmar_pass_vb = NULL;
static gpu_buffer_t *mmar_pass_ib = NULL;

// Inputs for the next pass, in the order the pass declares them. A pass samples
// the results of earlier ones, so without this every chained pass reads an
// unbound texture -- which draws without complaint and produces nothing.
// Rendered pass targets, keyed by pass id. The material shader samples these
// by name, and uniforms_ext_tex_link resolves "_mmar_<id>" through here.
any_map_t *mmar_pass_targets = NULL;

// Exposed parameter values, by stable id. One store serves both halves: the
// kernel reaches them as host constants through "_mmar_p_<id>" links, and a
// pass -- compiled standalone with its own parameter block -- has them written
// straight into its constant locations before it draws.
any_map_t *mmar_params = NULL;

static float mmar_param_get(char *id) {
	if (mmar_params == NULL) {
		return 0.0f;
	}
	float *v = any_map_get(mmar_params, id);
	return v == NULL ? 0.0f : *v;
}

void mmar_param_set(char *id, float value) {
	if (mmar_params == NULL) {
		mmar_params = any_map_create();
	}
	float *v = any_map_get(mmar_params, id);
	if (v == NULL) {
		v = malloc(sizeof(float));
		any_map_set(mmar_params, string_copy(id), v);
	}
	*v = value;
}

float mmar_param_value(char *id) {
	return mmar_param_get(id);
}

// Parameters the next pass declares, in the order its block declares them.
#define MMAR_MAX_PASS_PARAMS 32
static char *mmar_pass_params[MMAR_MAX_PASS_PARAMS];
static int   mmar_pass_param_count = 0;

void mmar_add_pass_param(char *id) {
	if (mmar_pass_param_count >= MMAR_MAX_PASS_PARAMS) {
		console_error("mmar: too many parameters for one pass");
		return;
	}
	mmar_pass_params[mmar_pass_param_count++] = string_copy(id);
}

#define MMAR_MAX_INPUTS 16
static gpu_texture_t *mmar_pass_inputs[MMAR_MAX_INPUTS];
static int            mmar_pass_input_count = 0;

void mmar_bind_pass_input(gpu_texture_t *tex) {
	if (mmar_pass_input_count >= MMAR_MAX_INPUTS) {
		console_error("mmar: too many inputs for one pass");
		return;
	}
	mmar_pass_inputs[mmar_pass_input_count++] = tex;
}

static void mmar_pass_create_quad(void) {
	// One oversized triangle rather than two triangles: no shared edge to crack
	// along, and one less vertex to transform.
	gpu_vertex_structure_t *structure = ALLOC_INIT(gpu_vertex_structure_t, {0});
	gpu_vertex_structure_add(structure, "pos", GPU_VERTEX_DATA_F32_2X);

	mmar_pass_vb   = gpu_create_vertex_buffer(3, structure);
	float *vertices = gpu_vertex_buffer_lock(mmar_pass_vb);
	vertices[0] = -1.0f; vertices[1] = -1.0f;
	vertices[2] =  3.0f; vertices[3] = -1.0f;
	vertices[4] = -1.0f; vertices[5] =  3.0f;
	gpu_vertex_buffer_unlock(mmar_pass_vb);

	mmar_pass_ib  = gpu_create_index_buffer(3);
	uint32_t *idx = gpu_index_buffer_lock(mmar_pass_ib);
	idx[0] = 0; idx[1] = 1; idx[2] = 2;
	gpu_index_buffer_unlock(mmar_pass_ib);
}

// NOTE: gpu_create_shaders_from_kong below blocks when this is called from a
// plugin script under --background. Measured by bisection: stopping before the
// call completes the import normally, stopping immediately after it hangs. So
// the runtime kong compiler needs a context the headless script path does not
// provide, and passes have to be driven from inside the render path rather than
// at import time. See issue #13.
// Write parameter values into a shader source as literals.
//
// Longest id first. `constants.seed_fbm4` is a prefix of
// `constants.seed_fbm4_2`, and replacing the short one first leaves `_2`
// dangling on the end of a number -- which kong reports as a missing bracket
// three lines later. The cooker orders its own renames the same way.
//
// Fixed point, never an exponent: kong tells int from float by the decimal
// point alone and has no exponent literals. UPSTREAM-QUIRKS.md.
static char *mmar_bake_params(char *src, char **ids, int count) {
	if (src == NULL || ids == NULL || count <= 0) {
		return src;
	}
	bool *done = calloc(count, sizeof(bool));
	for (int k = 0; k < count; ++k) {
		int best = -1;
		int best_len = -1;
		for (int i = 0; i < count; ++i) {
			if (done[i] || ids[i] == NULL) {
				continue;
			}
			int l = string_length(ids[i]);
			if (l > best_len) {
				best     = i;
				best_len = l;
			}
		}
		if (best < 0) {
			break;
		}
		done[best] = true;
		src        = string_replace_all(src, string("constants.%s", ids[best]),
		                                string("%.9f", mmar_param_get(ids[best])));
	}
	free(done);
	return src;
}

gpu_texture_t *mmar_run_pass(char *id, char *kong_source, char *format, int size) {
	if (kong_source == NULL || size <= 0) {
		console_error("mmar: pass has no source, or a size of zero");
		return NULL;
	}
	if (mmar_pass_vb == NULL) {
		mmar_pass_create_quad();
	}

	char *fmt = (format == NULL || string_length(format) == 0) ? "RGBA32" : format;

	shader_context_t *con = ALLOC_INIT(shader_context_t,
	                                   {.name              = "mmar_pass",
	                                    .depth_write       = false,
	                                    .compare_mode      = "always",
	                                    .cull_mode         = "none",
	                                    .shader_from_source = true,
	                                    .bind_textures     = any_array_create_from_raw((void *[]){}, 0),
	                                    .vertex_elements   = any_array_create_from_raw(
                                          (void *[]){
                                              ALLOC_INIT(vertex_element_t, {.name = "pos", .data = "float2"}),
                                          },
                                          1),
	                                    .color_attachments = any_array_create_from_raw((void *[]){fmt}, 1)});
	con->_ = ALLOC_INIT(shader_context_runtime_t, {0});

	// Parameter values go into the source, not into a bound constant.
	//
	// Declaring them resolved a location for every parameter and the value
	// written to it never reached the shader -- see UPSTREAM-QUIRKS.md, which
	// is also why the kernel bakes its own. A pass is compiled from source on
	// every run, so there is nothing to keep by binding.
	char *src = mmar_bake_params(kong_source, mmar_pass_params, mmar_pass_param_count);

	gpu_create_shaders_from_kong(src, &con->vertex_shader, &con->fragment_shader, &con->_->vertex_shader_size,
	                             &con->_->fragment_shader_size);
	// On macOS gpu_create_shaders_from_kong returns the Metal source but leaves
	// the sizes alone -- only the SPIR-V branch writes them. gpu_shader_init
	// reads the entry name out of the first line of that source, so a length of
	// zero yields an empty name and a nil vertex function.
	if (con->vertex_shader != NULL && con->_->vertex_shader_size == 0) {
		con->_->vertex_shader_size = string_length(con->vertex_shader);
	}
	if (con->fragment_shader != NULL && con->_->fragment_shader_size == 0) {
		con->_->fragment_shader_size = string_length(con->fragment_shader);
	}
	if (con->vertex_shader == NULL || con->fragment_shader == NULL || con->_->vertex_shader_size == 0) {
		console_error("mmar: pass did not compile");
		return NULL;
	}
	shader_context_load(con);


	gpu_texture_t *target = gpu_create_render_target(size, size, shader_context_get_tex_format(fmt));

	_gpu_begin(target, NULL, NULL, GPU_CLEAR_COLOR, 0, 0.0);
	gpu_set_pipeline(con->_->pipe);
	for (int i = 0; i < mmar_pass_input_count; ++i) {
		gpu_set_texture(i, mmar_pass_inputs[i]);
	}
	gpu_set_vertex_buffer(mmar_pass_vb);
	gpu_set_index_buffer(mmar_pass_ib);
	gpu_draw();
	gpu_end();

	mmar_pass_input_count = 0;
	mmar_pass_param_count = 0;
	if (id != NULL) {
		if (mmar_pass_targets == NULL) {
			mmar_pass_targets = any_map_create();
		}
		// string_copy: the id came from a map the caller decoded and will drop,
		// and this key is looked up later when a material shader binds the
		// target by name.
		any_map_set(mmar_pass_targets, string_copy(id), target);
	}
	return target;
}

// Make an imported archive appear in the material node editor's Add menu.
//
// plugin_material_custom_nodes_set maps a node type to the function that emits
// its shader, but nothing puts that type in front of the user: the Add menu is
// built from nodes_material_list, and a plugin can only contribute to it by
// registering a category. Without this an imported .mmar is a node type nobody
// can place.
//
// Built here rather than in the plugin because a ui_node_t definition is a
// nested structure of sockets and defaults, and assembling one through MiniC is
// considerably more error-prone than calling a function that does it.
// The node definition, kept so parameters can be added to it after the fact.
// The archive is walked in one pass: the kernel is registered before its
// parameters are known, so the sockets are appended as they turn up.
static ui_node_t     *mmar_node_def       = NULL;
static string_array_t *mmar_node_param_ids = NULL;

void mmar_register_node(char *name) {
	// Copy: name is a MiniC value and does not survive the call that passed it.
	// It is stored in the node definition and used as the node type forever
	// after, so a borrowed pointer leaves a node with no name and no type --
	// which draws as an empty box and never matches in parser_material.
	name = string_copy(name);
	ui_node_t *def = ALLOC_INIT(ui_node_t,
	                            {.id     = 0,
	                             .name   = name,
	                             .type   = name, // matches plugin_material_custom_nodes_set
	                             .x      = 0,
	                             .y      = 0,
	                             .color  = 0xff4982a0,
	                             .inputs = any_array_create_from_raw((void *[]){}, 0),
	                             .outputs =
	                                 any_array_create_from_raw((void *[]){
	                                                               ALLOC_INIT(ui_node_socket_t, {.id            = 0,
	                                                                                             .node_id       = 0,
	                                                                                             .name          = "Color",
	                                                                                             .type          = "RGBA",
	                                                                                             .color         = 0xffc7c729,
	                                                                                             .default_value = f32_array_create_xyzw(0.8, 0.8, 0.8, 1.0),
	                                                                                             .min           = 0.0,
	                                                                                             .max           = 1.0,
	                                                                                             .precision     = 100,
	                                                                                             .display       = 0}),
	                                                           },
	                                                           1),
	                             .buttons = any_array_create_from_raw((void *[]){}, 0),
	                             .width   = 0,
	                             .flags   = 0});

	any_array_t *list = any_array_create_from_raw((void *[]){def}, 1);
	plugin_material_category_add("mmar", list);
	mmar_node_def       = def;
	mmar_node_param_ids = string_array_create(0);
}

// Give the node one knob per exposed control.
//
// Without this the node carries a single Color output and nothing else, which
// is what it looked like from the UI: the parameters were in the archive, were
// spliced into the shader as host constants, and had no way to be reached.
// The socket name is the parameter id, since that is what the kernel reads and
// what a value written here has to land on.
void mmar_add_node_param(char *id, float value, float min, float max) {
	if (mmar_node_def == NULL || mmar_node_param_ids == NULL) {
		return;
	}
	if (max <= min) {
		// No range in the archive. Leave room on both sides of the cooked
		// value rather than clamping it to a guess.
		min = value < 0.0f ? value * 2.0f - 1.0f : 0.0f;
		max = value > 0.0f ? value * 2.0f + 1.0f : 1.0f;
	}
	char *stable = string_copy(id);
	any_array_push(mmar_node_def->inputs,
	               ALLOC_INIT(ui_node_socket_t, {.id            = 0,
	                                             .node_id       = 0,
	                                             .name          = stable,
	                                             .type          = "VALUE",
	                                             .color         = 0xffa1a1a1,
	                                             .default_value = f32_array_create_x(value),
	                                             .min           = min,
	                                             .max           = max,
	                                             .precision     = 100,
	                                             .display       = 0}));
	string_array_push(mmar_node_param_ids, stable);
}

// Read the knobs off the node instance being parsed, in the order the sockets
// were added. parser_material hands the custom-node callback the instance, so
// this is where a value the user turned becomes the value the kernel reads --
// every rebuild of the material, which is what makes the knob live.
void mmar_read_node_params(void *node) {
	ui_node_t *n = (ui_node_t *)node;
	if (n == NULL || n->inputs == NULL || mmar_node_param_ids == NULL) {
		return;
	}
	i32 count = n->inputs->length;
	if (count > mmar_node_param_ids->length) {
		count = mmar_node_param_ids->length;
	}
	for (i32 i = 0; i < count; ++i) {
		ui_node_socket_t *sock = n->inputs->buffer[i];
		if (sock == NULL || sock->default_value == NULL || sock->default_value->length < 1) {
			continue;
		}
		mmar_param_set(mmar_node_param_ids->buffer[i], sock->default_value->buffer[0]);
	}
}

// A material node that samples by UV has to ask for the tex vertex element, or
// tex_coord is never plumbed through the vertex shader and every sample lands
// at the same point. ArmorPaint's own texture nodes do this via
// node_shader_context_add_elem; that takes the shader's context, which is not
// something MiniC can reach through an opaque pointer, so this wraps it.
void mmar_need_tex_coord(void) {
	if (parser_material_kong == NULL) {
		return;
	}
	node_shader_context_add_elem(parser_material_kong->context, "tex", "short2norm");
}

// Kernel state, held here rather than in the plugin.
//
// MiniC values do not survive the call that made them. Measured directly: a
// 29366-byte kernel source stored into a map reads back at 29366 bytes inside
// the importing call and 0 bytes on the next frame, with string_copy making no
// difference. So a plugin cannot hold anything between being handed an archive
// and being asked to build a shader from it -- which is every frame the
// material is rebuilt, and which presented as "no kernel registered".
static char           *mmar_kernel_source = NULL;
static char           *mmar_kernel_entry  = NULL;
static string_array_t *mmar_kernel_reads  = NULL;
static string_array_t *mmar_kernel_params = NULL;

void mmar_set_kernel(char *source, char *entry) {
	mmar_kernel_source = string_copy(source);
	mmar_kernel_entry  = string_copy(entry);
	mmar_kernel_reads  = string_array_create(0);
	mmar_kernel_params = string_array_create(0);
}

void mmar_add_kernel_param(char *id) {
	if (mmar_kernel_params == NULL) {
		mmar_kernel_params = string_array_create(0);
	}
	string_array_push(mmar_kernel_params, string_copy(id));
}

// How much kernel is currently held. Exists because the plugin-side store
// looked fine at import and was empty a frame later, and that is worth being
// able to check rather than assume.
int mmar_kernel_bytes(void) {
	return mmar_kernel_source == NULL ? 0 : (int)strlen(mmar_kernel_source);
}

// How many pass inputs the kernel will declare when spliced. Worth being able
// to check: registering the kernel clears this list, so reads recorded before
// that call vanish and the shader then samples textures it never declared.
int mmar_kernel_read_count(void) {
	return mmar_kernel_reads == NULL ? 0 : mmar_kernel_reads->length;
}

void mmar_add_kernel_read(char *id) {
	if (mmar_kernel_reads == NULL) {
		mmar_kernel_reads = string_array_create(0);
	}
	string_array_push(mmar_kernel_reads, string_copy(id));
}

// Splice the kernel into the shader being assembled and return the expression
// for the node's socket. Done in one call so the plugin holds no state at all.
char *mmar_splice_kernel(void) {
	if (mmar_kernel_source == NULL || mmar_kernel_entry == NULL || parser_material_kong == NULL) {
		return "float3(0.0, 0.0, 0.0)";
	}
	// A node that samples by UV must ask for the tex vertex element, the way
	// image_texture_node.c does; without it tex_coord never reaches the vertex
	// shader and every sample reads the same point.
	node_shader_context_add_elem(parser_material_kong->context, "tex", "short2norm");

	// Each buffer pass the kernel samples, declared and linked to the target
	// rendered for it. uniforms_ext_tex_link resolves "_mmar_<id>".
	if (mmar_kernel_reads != NULL) {
		for (i32 i = 0; i < mmar_kernel_reads->length; ++i) {
			char *rid = mmar_kernel_reads->buffer[i];
			node_shader_add_texture(parser_material_kong, rid, string("_mmar_%s", rid));
		}
	}
	// Parameter values are written into the kernel as literals rather than
	// declared as bound constants.
	//
	// The bound route is the tidier one and it is already wired: each id has a
	// `_mmar_p_<id>` link and uniforms_ext_f32_link resolves it. It does not
	// reach the shader -- the same thing measured on the pass path, where a
	// location resolves per parameter and the value written to it never
	// arrives. Every node ArmorPaint ships takes the other route: it bakes its
	// values into the shader source and lets the material recompile, which
	// ui_nodes already does on every knob change. So does this.
	char *src = mmar_kernel_source;
	if (mmar_kernel_params != NULL) {
		src = mmar_bake_params(src, (char **)mmar_kernel_params->buffer, mmar_kernel_params->length);
	}
	node_shader_add_function(parser_material_kong, src);
	return string("%s(tex_coord)", mmar_kernel_entry);
}

// Register an imported archive as a material node, in one call.
//
// Both halves store the name: plugin_material_custom_nodes_set keys the parse
// callback on it, and the node definition carries it as its type. Both were
// being handed a MiniC pointer that is invalid the moment the importing call
// returns.
void mmar_register_material(char *name, void *parse_fn) {
	char *stable = string_copy(name);
	plugin_material_custom_nodes_set(stable, parse_fn);
	mmar_register_node(stable);
}

// Report what actually ended up in the Add menu, so a blank node can be told
// apart from a node that was never registered properly.
void mmar_node_debug(void) {
	if (nodes_material_categories == NULL || nodes_material_list == NULL) {
		console_info("mmar-debug: no node categories at all");
		return;
	}
	console_info(string("mmar-debug: %i categories, %i lists", nodes_material_categories->length, nodes_material_list->length));
	for (i32 i = 0; i < nodes_material_categories->length; ++i) {
		char *cat = (char *)nodes_material_categories->buffer[i];
		if (cat == NULL || !string_equals(cat, "mmar")) {
			continue;
		}
		ui_node_t_array_t *list = nodes_material_list->buffer[i];
		console_info(string("mmar-debug: category %i holds %i node(s)", i, list == NULL ? -1 : list->length));
		if (list != NULL && list->length > 0) {
			ui_node_t *n = list->buffer[0];
			console_info(string("mmar-debug: node name = %s", n->name == NULL ? "(null)" : n->name));
			console_info(string("mmar-debug: node type = %s", n->type == NULL ? "(null)" : n->type));
			console_info(string("mmar-debug: outputs = %i", n->outputs == NULL ? -1 : n->outputs->length));
			console_info(string("mmar-debug: inputs = %i", n->inputs == NULL ? -1 : n->inputs->length));
			if (n->inputs != NULL) {
				for (i32 s = 0; s < n->inputs->length; ++s) {
					ui_node_socket_t *sock = n->inputs->buffer[s];
					console_info(string("mmar-debug:   knob %s = %f (%f..%f)", sock->name,
					                    sock->default_value->buffer[0], sock->min, sock->max));
				}
			}
		}
	}
}

// Write a rendered pass to a PNG so its contents can actually be looked at.
//
// A pass that renders and a pass that renders nothing useful are
// indistinguishable from the outside: both report success and both leave the
// final material a flat colour. This is the only way to tell them apart
// without a GUI.
void mmar_dump_pass(char *id, char *path, int size) {
	if (mmar_pass_targets == NULL) {
		console_error("mmar: no pass targets to dump");
		return;
	}
	gpu_texture_t *t = any_map_get(mmar_pass_targets, id);
	if (t == NULL) {
		console_error(string("mmar: no target named %s", id));
		return;
	}
	buffer_t *b = buffer_create(size * size * 4);
	gpu_get_render_target_pixels(t, (uint8_t *)b->buffer);
	iron_write_png(path, b, size, size, 0);
	console_info(string("mmar: wrote %s", path));
}
