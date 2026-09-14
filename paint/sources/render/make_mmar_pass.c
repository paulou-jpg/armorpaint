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
gpu_texture_t *mmar_run_pass(char *kong_source, char *format, int size) {
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
	gpu_set_vertex_buffer(mmar_pass_vb);
	gpu_set_index_buffer(mmar_pass_ib);
	gpu_draw();
	gpu_end();

	return target;
}
