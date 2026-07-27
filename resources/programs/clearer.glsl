#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0, r8ui) uniform uimage3D occluder_texture;
uniform ivec3 min_cell;
uniform ivec3 max_cell;

void main()
{
    ivec3 global_voxel = ivec3(gl_GlobalInvocationID) + min_cell;
    if (any(greaterThanEqual(global_voxel, max_cell)))
        return;
    imageStore(occluder_texture, global_voxel, uvec4(0));
}
