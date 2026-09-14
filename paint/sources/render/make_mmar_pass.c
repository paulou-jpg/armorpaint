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

	gpu_create_shaders_from_kong(kong_source, &con->vertex_shader, &con->fragment_shader, &con->_->vertex_shader_size,
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
void mmar_register_node(char *name) {
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

void mmar_set_kernel(char *source, char *entry) {
	mmar_kernel_source = string_copy(source);
	mmar_kernel_entry  = string_copy(entry);
	mmar_kernel_reads  = string_array_create(0);
}

// How much kernel is currently held. Exists because the plugin-side store
// looked fine at import and was empty a frame later, and that is worth being
// able to check rather than assume.
int mmar_kernel_bytes(void) {
	return mmar_kernel_source == NULL ? 0 : (int)strlen(mmar_kernel_source);
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
	node_shader_add_function(parser_material_kong, mmar_kernel_source);
	return string("%s(tex_coord)", mmar_kernel_entry);
}
