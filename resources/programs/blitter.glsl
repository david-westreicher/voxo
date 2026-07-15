#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0) uniform usampler3D voxel_texture;
layout(binding = 1, r8ui) uniform uimage3D occluder_texture;

uniform mat4 obj_transform_inv;
ivec3 occluder_size = imageSize(occluder_texture);
ivec3 obj_dimensions = textureSize(voxel_texture, 0);

void main()
{
    ivec3 global_voxel = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(global_voxel, occluder_size)))
        return;

    vec3 global_pos = vec3(global_voxel) + 0.5;
    vec3 obj_pos = (obj_transform_inv * vec4(global_pos, 1.0)).xyz;
    ivec3 obj_voxel = ivec3(floor(obj_pos));
    if (any(lessThan(obj_voxel, ivec3(0))) || any(greaterThanEqual(obj_voxel, obj_dimensions)))
    {
        return;
    }
    uint voxel = texelFetch(voxel_texture, obj_voxel, 0).r;
    if (voxel == 0)
        return;
    imageStore(occluder_texture, global_voxel, uvec4(1));
}
