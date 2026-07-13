#version 430

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(binding = 0, r8ui) readonly uniform uimage3D src_mip;
layout(binding = 1, r8ui) writeonly uniform uimage3D dst_mip;

void main()
{
    ivec3 dst_size = imageSize(dst_mip);

    ivec3 dst = ivec3(gl_GlobalInvocationID);

    if (any(greaterThanEqual(dst, dst_size)))
        return;

    ivec3 src = dst * 2;

    uint occ = 0u;
    occ = max(occ, imageLoad(src_mip, src + ivec3(0, 0, 0)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(1, 0, 0)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(0, 1, 0)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(1, 1, 0)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(0, 0, 1)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(1, 0, 1)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(0, 1, 1)).r);
    occ = max(occ, imageLoad(src_mip, src + ivec3(1, 1, 1)).r);

    imageStore(dst_mip, dst, uvec4(occ));
}
