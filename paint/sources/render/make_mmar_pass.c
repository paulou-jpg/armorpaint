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

// Everything an imported archive owns.
//
// All of this used to be one set of globals, which worked for exactly one
// archive. Importing a second overwrote the first's kernel, appended its
// sockets to the same list and left its node pointing at the wrong material --
// so two imported materials both rendered whichever was imported last.
// parser_material hands the custom-node callback the node instance, and a
// node's type is the name it was registered under, so the right record is
// always reachable from the thing being parsed.
typedef struct mmar_material {
	char           *name;
	char           *archive;
	ui_node_t      *def;
	char           *kernel_source;
	char           *kernel_entry;
	string_array_t *kernel_reads;
	string_array_t *kernel_params;
	string_array_t *out_sockets;
	string_array_t *out_fields;
	string_array_t *node_param_ids;
	bool            passes_dirty;
} mmar_material_t;

static any_map_t       *mmar_materials = NULL; // name -> mmar_material_t *
static mmar_material_t *mmar_importing = NULL; // the one being imported
static mmar_material_t *mmar_rendering = NULL; // the one whose passes are running

static mmar_material_t *mmar_material_get(char *name) {
	if (name == NULL) {
		return NULL;
	}
	if (mmar_materials == NULL) {
		mmar_materials = any_map_create();
	}
	mmar_material_t *m = any_map_get(mmar_materials, name);
	if (m == NULL) {
		m       = ALLOC_INIT(mmar_material_t, {0});
		m->name = string_copy(name);
		any_map_set(mmar_materials, m->name, m);
	}
	return m;
}

// The material a node belongs to. A node's type is its registered name.
static mmar_material_t *mmar_material_for(void *node) {
	ui_node_t *n = (ui_node_t *)node;
	if (n == NULL || n->type == NULL || mmar_materials == NULL) {
		return mmar_importing;
	}
	mmar_material_t *m = any_map_get(mmar_materials, n->type);
	return m != NULL ? m : mmar_importing;
}

// Exposed parameter values, by stable id. One store serves both halves: the
// kernel reaches them as host constants through "_mmar_p_<id>" links, and a
// pass -- compiled standalone with its own parameter block -- has them written
// straight into its constant locations before it draws.
any_map_t *mmar_params = NULL;

static char *mmar_param_key(mmar_material_t *m, char *id);

static float mmar_param_get_m(mmar_material_t *m, char *id) {
	if (mmar_params == NULL) {
		return 0.0f;
	}
	float *v = any_map_get(mmar_params, mmar_param_key(m, id));
	return v == NULL ? 0.0f : *v;
}

// Bumped whenever a value actually moves, and recorded when the passes last
// rendered. A pass bakes its values in, so the only way a knob reaches one is
// to run it again -- and the only way to know that is needed is to notice the
// value changed. Comparing a counter beats walking the store every parse.
static uint32_t mmar_param_epoch          = 0;
static uint32_t mmar_param_epoch_rendered = 0;

// Scoped to the material. Two archives can both expose "hue".
static char *mmar_param_key(mmar_material_t *m, char *id) {
	return string("%s|%s", m == NULL || m->name == NULL ? "" : m->name, id);
}

static void mmar_param_set_m(mmar_material_t *m, char *id, float value);

// Import-time entry point: the plugin seeds defaults while an archive is being
// read, which is the one moment the material is unambiguous.
void mmar_param_set(char *id, float value) {
	mmar_param_set_m(mmar_importing, id, value);
}

float mmar_param_value(char *id) {
	return mmar_param_get_m(mmar_importing, id);
}

static void mmar_param_set_m(mmar_material_t *m, char *id, float value) {
	id = mmar_param_key(m, id);
	if (mmar_params == NULL) {
		mmar_params = any_map_create();
	}
	float *v = any_map_get(mmar_params, id);
	if (v == NULL) {
		v = malloc(sizeof(float));
		any_map_set(mmar_params, string_copy(id), v);
		*v = value;
		mmar_param_epoch++;
		return;
	}
	if (*v != value) {
		*v = value;
		mmar_param_epoch++;
	}
}

// The resolution a pass should render at, given the control that sizes it.
//
// A render target is allocated from this, so it is clamped to something a GPU
// will take and rounded to a power of two -- a knob drag passes through 1537
// on its way from 1024 to 2048, and nothing downstream wants a target that
// shape. Sizes are what a control drives here; the value is not written into
// the shader, the pass is re-rendered at it.
int mmar_pass_size_for(char *size_param, int fallback) {
	if (size_param == NULL) {
		return fallback;
	}
	float v = mmar_param_get_m(mmar_rendering, size_param);
	if (v < 16.0f) {
		v = 16.0f;
	}
	if (v > 4096.0f) {
		v = 4096.0f;
	}
	int size = 16;
	while (size < 4096 && (float)(size * 2) <= v * 1.5f) {
		size *= 2;
	}
	return size;
}

// Has any value moved since the passes were last rendered?
int mmar_params_changed(void) {
	return mmar_param_epoch != mmar_param_epoch_rendered ? 1 : 0;
}

void mmar_params_rendered(void) {
	mmar_param_epoch_rendered = mmar_param_epoch;
}

// The archive the passes come from, held here rather than in the plugin.
//
// Re-rendering happens a frame after the value changed, and a MiniC value does
// not survive the call that made it. The path is small and the archive is
// re-decoded from it each time -- the same trade e8b4143 made for the first
// render, now that there is more than one.
void mmar_set_archive(char *path) {
	if (mmar_importing != NULL) {
		mmar_importing->archive       = string_copy(path);
		mmar_importing->passes_dirty  = true;
	}
}

// The archive whose passes need rendering, and the material they belong to.
// Several can be waiting at once -- two imported materials whose controls both
// moved -- so this hands back one per call and clears it.
// Is another material still waiting for its passes? The frame callback renders
// one per call and registering it twice in a frame does not queue it twice, so
// whoever renders has to ask for another turn.
int mmar_has_dirty_passes(void) {
	if (mmar_materials == NULL) {
		return 0;
	}
	any_array_t *keys = map_keys(mmar_materials);
	for (i32 i = 0; i < keys->length; ++i) {
		mmar_material_t *m = any_map_get(mmar_materials, keys->buffer[i]);
		if (m != NULL && m->passes_dirty && m->archive != NULL) {
			return 1;
		}
	}
	return 0;
}

char *mmar_archive_path(void) {
	if (mmar_materials == NULL) {
		return NULL;
	}
	any_array_t *keys = map_keys(mmar_materials);
	for (i32 i = 0; i < keys->length; ++i) {
		mmar_material_t *m = any_map_get(mmar_materials, keys->buffer[i]);
		if (m != NULL && m->passes_dirty && m->archive != NULL) {
			m->passes_dirty = false;
			mmar_rendering  = m;
			return m->archive;
		}
	}
	return NULL;
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
static char *mmar_bake_params(mmar_material_t *m, char *src, char **ids, int count) {
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
		                                string("%.9f", mmar_param_get_m(m, ids[best])));
	}
	free(done);
	return src;
}

// What a pass keeps between runs.
//
// Compiling is the expensive half -- kong to Metal to pipeline, about as costly
// as every pass's draw put together -- and a pass re-renders whenever a control
// that drives one moves. The baked source is the cache key precisely because
// the values are in it: a pass with no parameters keeps its pipeline for the
// life of the import, and a parameterised one rebuilds only its own.
typedef struct mmar_pass_cache {
	char             *source; // the baked source this pipeline was built from
	shader_context_t *con;
	gpu_texture_t    *target;
	int               size;
	char             *format;
} mmar_pass_cache_t;

static any_map_t *mmar_pass_cache = NULL;

// Compiled vs reused since the counter was last read. A re-render that
// recompiled everything and one that recompiled the two passes a control
// actually drives look identical from the outside, and the difference is the
// whole reason the cache exists.
static int mmar_pass_compiled = 0;
static int mmar_pass_reused   = 0;

int mmar_pass_compile_count(void) {
	int n              = mmar_pass_compiled;
	mmar_pass_compiled = 0;
	return n;
}

// Milliseconds spent compiling and drawing since last read. Guessing which
// half of a re-render is slow is how you optimise the wrong one.
static double mmar_pass_ms_compile = 0.0;
static double mmar_pass_ms_draw    = 0.0;

int mmar_pass_compile_ms(void) {
	int n                = (int)mmar_pass_ms_compile;
	mmar_pass_ms_compile = 0.0;
	return n;
}

int mmar_pass_draw_ms(void) {
	int n             = (int)mmar_pass_ms_draw;
	mmar_pass_ms_draw = 0.0;
	return n;
}

// A stopwatch the plugin can wrap around work that happens on its side, so the
// archive decode can be told apart from the rendering it precedes.
static double mmar_mark_at = 0.0;

void mmar_mark(void) {
	mmar_mark_at = iron_time();
}

int mmar_elapsed_ms(void) {
	return (int)((iron_time() - mmar_mark_at) * 1000.0);
}

// Time spent splicing the kernel into the material shader: baking 16 values
// through a 49 KB source on every parse is the kind of thing that looks free
// and is not.
static double mmar_splice_ms_total = 0.0;

int mmar_splice_ms(void) {
	int n                = (int)mmar_splice_ms_total;
	mmar_splice_ms_total = 0.0;
	return n;
}

int mmar_pass_reuse_count(void) {
	int n            = mmar_pass_reused;
	mmar_pass_reused = 0;
	return n;
}

static void mmar_pass_draw(shader_context_t *con, gpu_texture_t *target, int size) {
	double t_draw = iron_time();
	_gpu_begin(target, NULL, NULL, GPU_CLEAR_COLOR, 0, 0.0);
	gpu_set_pipeline(con->_->pipe);
	for (int i = 0; i < mmar_pass_input_count; ++i) {
		gpu_set_texture(i, mmar_pass_inputs[i]);
	}
	// Between _gpu_begin and gpu_draw: gpu_draw unlocks this slot of the
	// constant ring, binds it and advances. A value written outside that
	// window lands in a slot no draw will read.
	uniforms_set_obj_consts(con, NULL);
	gpu_set_vertex_buffer(mmar_pass_vb);
	gpu_set_index_buffer(mmar_pass_ib);
	gpu_draw();
	gpu_end();
	mmar_pass_ms_draw += (iron_time() - t_draw) * 1000.0;
	mmar_pass_input_count = 0;
	mmar_pass_param_count = 0;
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

	// Parameter values are bound constants, not text substituted into the
	// source.
	//
	// They were baked because binding them did not work: a location resolved
	// for every parameter and the value written to it never arrived. The
	// locations were never the problem -- this pass resolves its three floats
	// to offsets 0, 4 and 8, which is exactly kong's packing. The write was.
	// gpu_set_float writes into whichever slot of the constant ring is
	// currently locked, and gpu_draw unlocks that slot, binds it, advances the
	// index and locks the next one. A value written outside that window goes
	// to a slot no draw will ever read, which is indistinguishable from a
	// parameter the author set to zero. So the write happens in
	// mmar_pass_draw, between _gpu_begin and gpu_draw. #15.
	char *src = kong_source;
	int   bound_count = mmar_pass_param_count;
	char *bound_ids[64];
	for (int bi = 0; bi < bound_count && bi < 64; ++bi) {
		bound_ids[bi] = mmar_pass_params[bi];
	}
	if (bound_count > 64) {
		bound_count = 64;
	}

	if (mmar_pass_cache == NULL) {
		mmar_pass_cache = any_map_create();
	}
	// Keyed by archive as well as id: pass ids are `texture_<n>` and start
	// again from the same numbers in the next archive, so two materials open
	// at once would otherwise share an entry.
	char              *key    = id == NULL ? NULL : string("%s|%s", mmar_archive_path(), id);
	mmar_pass_cache_t *cached = key == NULL ? NULL : any_map_get(mmar_pass_cache, key);
	bool reuse = cached != NULL && cached->con != NULL && cached->target != NULL &&
	             cached->size == size && string_equals(cached->format, fmt) &&
	             string_equals(cached->source, src);
	if (reuse) {
		// Nothing about this pass changed. It still redraws, because a pass
		// downstream of a changed one has to: its input moved even though it
		// did not. The draw is the cheap half.
		mmar_pass_reused++;
		mmar_pass_draw(cached->con, cached->target, size);
		return cached->target;
	}

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
	if (bound_count > 0) {
		con->constants = (shader_const_t_array_t *)any_array_create(0);
		for (int bi = 0; bi < bound_count; ++bi) {
			shader_const_t *sc = ALLOC_INIT(shader_const_t, {.name = string_copy(bound_ids[bi]),
			                                                 .type = "float",
			                                                 .link = string("_mmar_p_%s", bound_ids[bi])});
			any_array_push((any_array_t *)con->constants, sc);
		}
	}

	double t_compile = iron_time();
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
	mmar_pass_compiled++;
	mmar_pass_ms_compile += (iron_time() - t_compile) * 1000.0;

	// The target outlives the pipeline. Its size and format come from the
	// archive and do not move when a value does, so re-rendering reuses it --
	// otherwise a knob drag would allocate a fresh set every frame and the
	// material would keep sampling whichever one it was handed first.
	gpu_texture_t *target = cached != NULL ? cached->target : NULL;
	if (target == NULL || (cached != NULL && (cached->size != size || !string_equals(cached->format, fmt)))) {
		if (target != NULL) {
			gpu_texture_destroy(target);
		}
		target = gpu_create_render_target(size, size, shader_context_get_tex_format(fmt));
	}

	mmar_pass_draw(con, target, size);

	if (id != NULL) {
		if (mmar_pass_targets == NULL) {
			mmar_pass_targets = any_map_create();
		}
		// string_copy: the id came from a map the caller decoded and will drop,
		// and this key is looked up later when a material shader binds the
		// target by name.
		any_map_set(mmar_pass_targets, string_copy(id), target);

		if (cached == NULL) {
			cached = ALLOC_INIT(mmar_pass_cache_t, {0});
			any_map_set(mmar_pass_cache, string_copy(key), cached);
		}
		else if (cached->con != NULL) {
			// Replaced, not accumulated: one pipeline per pass at a time,
			// however many times a control moves.
			shader_context_delete(cached->con);
		}
		cached->source = string_copy(src);
		cached->con    = con;
		cached->target = target;
		cached->size   = size;
		cached->format = string_copy(fmt);
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
// "Connect to Outputs": wire every socket to the material output of the same
// name, in one click.
//
// The node offers nine channels and the output node takes nine inputs under
// exactly the same names, so the mapping is not a judgement -- it is a join,
// and doing it by hand nine times per material is the kind of work a button
// exists for.
// Defined in ui/ui_nodes.c, which the unity build includes after this file.
extern bool ui_nodes_recompile_mat;
extern bool ui_nodes_recompile_mat_final;

static void mmar_connect_button(i32 node_id) {
	if (!ui_button(tr("Connect to Outputs"), UI_ALIGN_CENTER, "")) {
		return;
	}
	ui_node_canvas_t *canvas = ui_nodes_get_canvas(true);
	if (canvas == NULL) {
		return;
	}
	ui_node_t *node = ui_get_node(canvas->nodes, node_id);
	ui_node_t *out  = parser_material_node_by_type(canvas->nodes, "OUTPUT_MATERIAL_PBR");
	if (node == NULL || out == NULL || node->outputs == NULL || out->inputs == NULL) {
		console_error("mmar: no material output node to connect to");
		return;
	}

	int made = 0;
	for (i32 i = 0; i < node->outputs->length; ++i) {
		ui_node_socket_t *src = node->outputs->buffer[i];
		for (i32 j = 0; j < out->inputs->length; ++j) {
			ui_node_socket_t *dst = out->inputs->buffer[j];
			if (src == NULL || dst == NULL || src->name == NULL || dst->name == NULL ||
			    !string_equals(src->name, dst->name)) {
				continue;
			}
			// One link per input, the rule a dragged connection follows.
			for (i32 k = canvas->links->length - 1; k >= 0; --k) {
				ui_node_link_t *l = canvas->links->buffer[k];
				if (l == NULL || l->to_id != out->id || l->to_socket != j) {
					continue;
				}
				for (i32 z = k; z < canvas->links->length - 1; ++z) {
					canvas->links->buffer[z] = canvas->links->buffer[z + 1];
				}
				canvas->links->length--;
			}
			ui_node_link_t *l = (ui_node_link_t *)malloc(sizeof(ui_node_link_t));
			l->id             = ui_next_link_id(canvas->links);
			l->from_id        = node->id;
			l->from_socket    = i;
			l->to_id          = out->id;
			l->to_socket      = j;
			any_array_push(canvas->links, l);
			made++;
			break;
		}
	}
	console_info(string("mmar: connected %i socket(s) to the material output", made));
	ui_nodes_recompile_mat       = true;
	ui_nodes_recompile_mat_final = true;
}

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
	                             .buttons = any_array_create_from_raw(
	                                 (void *[]){
	                                     ALLOC_INIT(ui_node_button_t, {.name          = "mmar_connect_button",
	                                                                   .type          = "CUSTOM",
	                                                                   .output        = -1,
	                                                                   .default_value = f32_array_create_x(0),
	                                                                   .data          = NULL,
	                                                                   .min           = 0.0,
	                                                                   .max           = 1.0,
	                                                                   .precision     = 100,
	                                                                   .height        = 1}),
	                                 },
	                                 1),
	                             .width   = 0,
	                             .flags   = 0});

	// One category, however many archives are imported. category_add appends,
	// so calling it per import left N categories holding one node each --
	// which is what the Add menu showed.
	any_array_t *list = NULL;
	if (nodes_material_categories != NULL && nodes_material_list != NULL) {
		for (i32 i = 0; i < nodes_material_categories->length; ++i) {
			char *cat = (char *)nodes_material_categories->buffer[i];
			if (cat != NULL && string_equals(cat, "mmar")) {
				list = nodes_material_list->buffer[i];
				break;
			}
		}
	}
	if (list == NULL) {
		plugin_material_category_add("mmar", any_array_create_from_raw((void *[]){def}, 1));
	}
	else {
		// Re-importing the same archive replaces its entry rather than adding
		// a second one with the same name.
		bool replaced = false;
		for (i32 i = 0; i < list->length; ++i) {
			ui_node_t *existing = list->buffer[i];
			if (existing != NULL && existing->type != NULL && string_equals(existing->type, name)) {
				list->buffer[i] = def;
				replaced        = true;
				break;
			}
		}
		if (!replaced) {
			any_array_push(list, def);
		}
	}

	if (ui_nodes_custom_buttons != NULL) {
		any_map_set(ui_nodes_custom_buttons, "mmar_connect_button", mmar_connect_button);
	}

	mmar_material_t *m = mmar_material_get(name);
	m->def             = def;
	m->node_param_ids  = string_array_create(0);
	m->out_sockets     = string_array_create(0);
	m->out_fields      = string_array_create(0);
	mmar_importing     = m;
	// Deliberately not touching kernel_source here: set_kernel runs first now,
	// and clearing it would put back the bug this pair was fixed for.
}

// Give the node one socket per channel the kernel produces.
//
// It shipped with a single Color output, which is all a host could ask for --
// while mmar_eval computed the normal, the occlusion, the roughness and the
// rest on its way to albedo and dropped them. The entry point returns a struct
// now, so a socket is a field read off one evaluation rather than another run
// of the whole graph.
void mmar_add_node_output(char *socket, char *field, char *socket_type) {
	mmar_material_t *m = mmar_importing;
	if (m == NULL || m->def == NULL) {
		return;
	}
	if (m->out_sockets->length == 0) {
		// The placeholder Color socket goes when the real ones arrive. Per
		// material: this used to test a shared list, so the second archive
		// imported kept its placeholder and came out with ten sockets.
		m->def->outputs->length = 0;
	}
	char *name  = string_copy(socket);
	bool  value = string_equals(socket_type, "VALUE");
	any_array_push(m->def->outputs,
	               ALLOC_INIT(ui_node_socket_t,
	                          {.id            = 0,
	                           .node_id       = 0,
	                           .name          = name,
	                           .type          = string_copy(socket_type),
	                           .color         = value ? 0xffa1a1a1 : 0xffc7c729,
	                           .default_value = value ? f32_array_create_x(0.0)
	                                                  : f32_array_create_xyzw(0.0, 0.0, 0.0, 1.0),
	                           .min           = 0.0,
	                           .max           = 1.0,
	                           .precision     = 100,
	                           .display       = 0}));
	string_array_push(m->out_sockets, name);
	string_array_push(m->out_fields, string_copy(field));
}

// Give the node one knob per exposed control.
//
// Without this the node carries a single Color output and nothing else, which
// is what it looked like from the UI: the parameters were in the archive, were
// spliced into the shader as host constants, and had no way to be reached.
// The socket name is the parameter id, since that is what the kernel reads and
// what a value written here has to land on.
void mmar_add_node_param(char *id, float value, float min, float max) {
	mmar_material_t *m = mmar_importing;
	if (m == NULL || m->def == NULL) {
		return;
	}
	if (max <= min) {
		// No range in the archive. Leave room on both sides of the cooked
		// value rather than clamping it to a guess.
		min = value < 0.0f ? value * 2.0f - 1.0f : 0.0f;
		max = value > 0.0f ? value * 2.0f + 1.0f : 1.0f;
	}
	char *stable = string_copy(id);
	any_array_push(m->def->inputs,
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
	string_array_push(m->node_param_ids, stable);
}

// Read the knobs off the node instance being parsed, in the order the sockets
// were added. parser_material hands the custom-node callback the instance, so
// this is where a value the user turned becomes the value the kernel reads --
// every rebuild of the material, which is what makes the knob live.
void mmar_read_node_params(void *node) {
	ui_node_t       *n = (ui_node_t *)node;
	mmar_material_t *m = mmar_material_for(node);
	if (n == NULL || n->inputs == NULL || m == NULL || m->node_param_ids == NULL) {
		return;
	}
	i32 count = n->inputs->length;
	if (count > m->node_param_ids->length) {
		count = m->node_param_ids->length;
	}
	for (i32 i = 0; i < count; ++i) {
		ui_node_socket_t *sock = n->inputs->buffer[i];
		if (sock == NULL || sock->default_value == NULL || sock->default_value->length < 1) {
			continue;
		}
		mmar_param_set_m(m, m->node_param_ids->buffer[i], sock->default_value->buffer[0]);
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
// Held per material rather than once: a second import used to overwrite the
// first's kernel, so both nodes rendered whichever archive was read last.
// Takes the name and selects the material itself, rather than relying on a
// registration call having happened first.
//
// Ordering mattered and was invisible: kernel state is per material, and the
// call that said which material was the registration -- so setting the kernel
// first stored it against nothing and the node spliced its fallback. Doing it
// the other way round was worse: MiniC values do not survive the call that
// passed them, and `entry` came back as garbage on the second import, so the
// copy would have been garbage too. Both arguments are consumed here, first,
// before anything else can invalidate them.
void mmar_set_kernel(char *name, char *source, char *entry) {
	mmar_material_t *m = mmar_material_get(name);
	if (m == NULL) {
		console_error("mmar: no material to attach a kernel to");
		return;
	}
	mmar_importing   = m;
	m->kernel_source = string_copy(source);

	// The entry name comes back corrupted on a second import in one session --
	// the 50 KB source survives and the ten-byte name does not. Not diagnosed;
	// what matters is that an unusable name splices `<garbage>(tex_coord)` and
	// the material renders flat with every step reporting success. A name that
	// is not an identifier is refused and the format's own entry point used,
	// out loud.
	bool ok = entry != NULL && entry[0] != '\0' && !(entry[0] >= '0' && entry[0] <= '9');
	for (const char *c = entry; ok && *c != '\0'; ++c) {
		ok = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') || (*c >= '0' && *c <= '9') || *c == '_';
	}
	if (!ok) {
		console_error(string("mmar: unusable kernel entry for '%s'; using mmar_eval", name));
		entry = "mmar_eval";
	}
	m->kernel_entry = string_copy(entry);
	m->kernel_reads  = string_array_create(0);
	m->kernel_params = string_array_create(0);
}

// The entry name as held, for reporting. The caller's copy may already be gone.
char *mmar_kernel_entry_name(void) {
	return mmar_importing == NULL || mmar_importing->kernel_entry == NULL
	           ? "(none)"
	           : mmar_importing->kernel_entry;
}

void mmar_add_kernel_param(char *id) {
	if (mmar_importing == NULL) {
		return;
	}
	if (mmar_importing->kernel_params == NULL) {
		mmar_importing->kernel_params = string_array_create(0);
	}
	string_array_push(mmar_importing->kernel_params, string_copy(id));
}

// How much kernel is currently held. Exists because the plugin-side store
// looked fine at import and was empty a frame later, and that is worth being
// able to check rather than assume.
int mmar_kernel_bytes(void) {
	return mmar_importing == NULL || mmar_importing->kernel_source == NULL
	           ? 0
	           : (int)strlen(mmar_importing->kernel_source);
}

// How many pass inputs the kernel will declare when spliced. Worth being able
// to check: registering the kernel clears this list, so reads recorded before
// that call vanish and the shader then samples textures it never declared.
int mmar_kernel_read_count(void) {
	return mmar_importing == NULL || mmar_importing->kernel_reads == NULL
	           ? 0
	           : mmar_importing->kernel_reads->length;
}

void mmar_add_kernel_read(char *id) {
	if (mmar_importing == NULL) {
		return;
	}
	if (mmar_importing->kernel_reads == NULL) {
		mmar_importing->kernel_reads = string_array_create(0);
	}
	string_array_push(mmar_importing->kernel_reads, string_copy(id));
}

// Splice the kernel into the shader being assembled and return the expression
// for the node's socket. Done in one call so the plugin holds no state at all.
char *mmar_splice_kernel(void *node, char *socket) {
	mmar_material_t *m = mmar_material_for(node);
	if (m == NULL || m->kernel_source == NULL || m->kernel_entry == NULL || parser_material_kong == NULL) {
		return "float3(0.0, 0.0, 0.0)";
	}
	double t_splice = iron_time();
	// A node that samples by UV must ask for the tex vertex element, the way
	// image_texture_node.c does; without it tex_coord never reaches the vertex
	// shader and every sample reads the same point.
	node_shader_context_add_elem(parser_material_kong->context, "tex", "short2norm");

	// Each buffer pass the kernel samples, declared and linked to the target
	// rendered for it. uniforms_ext_tex_link resolves "_mmar_<id>".
	if (m->kernel_reads != NULL) {
		for (i32 i = 0; i < m->kernel_reads->length; ++i) {
			char *rid = m->kernel_reads->buffer[i];
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
	char *src = m->kernel_source;
	if (m->kernel_params != NULL) {
		src = mmar_bake_params(m, src, (char **)m->kernel_params->buffer, m->kernel_params->length);
	}
	node_shader_add_function(parser_material_kong, src);
	mmar_splice_ms_total += (iron_time() - t_splice) * 1000.0;

	// Which field this socket reads. An archive cooked before the entry point
	// returned a struct has no outputs, and its kernel still returns albedo
	// directly -- so the bare call is the right expression for it.
	if (m->out_sockets == NULL || m->out_sockets->length == 0) {
		return string("%s(tex_coord)", m->kernel_entry);
	}
	char *field = m->out_fields->buffer[0];
	for (i32 i = 0; i < m->out_sockets->length; ++i) {
		if (socket != NULL && string_equals(m->out_sockets->buffer[i], socket)) {
			field = m->out_fields->buffer[i];
			break;
		}
	}
	return string("%s(tex_coord).%s", m->kernel_entry, field);
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
		for (i32 k = 0; list != NULL && k < list->length; ++k) {
			ui_node_t *n = list->buffer[k];
			console_info(string("mmar-debug: node name = %s", n->name == NULL ? "(null)" : n->name));
			console_info(string("mmar-debug: node type = %s", n->type == NULL ? "(null)" : n->type));
			console_info(string("mmar-debug: outputs = %i", n->outputs == NULL ? -1 : n->outputs->length));
			if (n->outputs != NULL) {
				for (i32 s = 0; s < n->outputs->length; ++s) {
					ui_node_socket_t *sock = n->outputs->buffer[s];
					console_info(string("mmar-debug:   socket %s : %s", sock->name, sock->type));
				}
			}
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
