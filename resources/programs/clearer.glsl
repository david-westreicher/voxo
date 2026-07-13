#version 450

#define GLOBAL_DIMENSIONS ivec3(1,1,1)

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0, r8ui) uniform uimage3D occluder_texture;

void main()
{
    ivec3 global_voxel = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(global_voxel, GLOBAL_DIMENSIONS)))
        return;
    imageStore(occluder_texture, global_voxel, uvec4(0));
}
