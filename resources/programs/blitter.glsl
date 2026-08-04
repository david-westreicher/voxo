#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0) uniform usampler3D voxel_texture;
layout(binding = 1, r8ui) uniform uimage3D occluder_texture;
layout(binding = 2) uniform sampler2D material_texture;
uniform mat4 obj_transform_inv;
uniform ivec3 min_cell; // relative to occluder origin
uniform ivec3 max_cell; // relative to occluder origin
uniform ivec3 occluder_translation;
uniform int material_row;

ivec3 obj_dimensions = textureSize(voxel_texture, 0);

void main()
{
    ivec3 global_voxel = ivec3(gl_GlobalInvocationID) + min_cell;
    if (any(lessThan(global_voxel, min_cell)) || any(greaterThanEqual(global_voxel, max_cell)))
        return;

    vec3 global_pos = vec3(global_voxel + occluder_translation) + 0.5;
    vec3 obj_pos = (obj_transform_inv * vec4(global_pos, 1.0)).xyz;
    ivec3 obj_voxel = ivec3(floor(obj_pos));
    if (any(lessThan(obj_voxel, ivec3(0))) || any(greaterThanEqual(obj_voxel, obj_dimensions)))
    {
        return;
    }
    uint voxel_material = texelFetch(voxel_texture, obj_voxel, 0).r;
    if (voxel_material == 0)
        return;
    ivec2 material_coord = ivec2(voxel_material, material_row);
    float transparency = max(0, -texelFetch(material_texture, material_coord, 0).a);
    if (transparency > 0.0) {
        return;
    }
    imageStore(occluder_texture, global_voxel, uvec4(1));
}
