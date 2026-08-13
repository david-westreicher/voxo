#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#include programs/utils.glsl
#line 18

in vec2 uv;

uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform usampler3D occluder_texture;
uniform ivec3 occluder_translation;

ivec3 size = textureSize(occluder_texture, 0);
Box bbox = Box(vec3(0.0), vec3(size));
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

out vec4 fragColor;

bool voxelmap(vec3 p, Box bbox, usampler3D u_voxel_data)
{
    vec3 local_coord = (p + 0.5) / (bbox.max - bbox.min);
    return textureLod(u_voxel_data, local_coord, 0.0).r > 0u;
}

Hit dda(Ray ray, int max_steps, usampler3D voxels, Box bbox) {
    vec3 pos = ray.origin;
    vec3 rayDir = ray.direction;
    Hit hit;
    hit.hit = false;

    vec3 map = floor(pos);
    vec3 stepDir = vec3(0);
    vec3 sideDist = vec3(9e9);
    vec3 deltaDist = 1. / abs(rayDir);
    float side = 0.;
    vec3 S = step(0., rayDir);

    stepDir = 2. * S - 1.;
    sideDist = (S - stepDir * fract(pos)) * deltaDist;

    bool has_entered = is_inside_box(map, bbox);
    int i;
    for (i = 0; i < max_steps; i++) {
        vec4 conds = step(sideDist.xxyy, sideDist.yzzx);
        vec3 cases = vec3(0);
        cases.x = conds.x * conds.y;
        cases.y = (1. - cases.x) * conds.z * conds.w;
        cases.z = (1. - cases.x) * (1. - cases.y);
        sideDist += max((2. * cases - 1.) * deltaDist, 0.);
        map += cases * stepDir;
        if (is_inside_box(map, bbox) && !has_entered) {
            has_entered = true;
        }
        if (has_entered && !is_inside_box(map, bbox)) {
            return hit;
        }
        if (has_entered && voxelmap(map, bbox, voxels)) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }
    if (!has_entered || i == max_steps) {
        return hit;
    }
    vec3 normal = vec3(0.0);
    normal[int(side)] = -1. * sign(rayDir[int(side)]); // voxel face debug
    vec3 p = map + .5 - stepDir * .5; // Point on axis plane
    float t = (dot(normal, p - pos)) / dot(normal, rayDir);

    hit.hit = true;
    hit.t = t;
    hit.position = pos + rayDir * t;
    hit.voxel = map;
    hit.normal = normal;
    return hit;
}

void main() {
    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, vec2(0.0));
    camera_ray.origin -= occluder_translation;
    Hit hit = dda(camera_ray, MAX_STEPS, occluder_texture, bbox);
    if (!hit.hit) {
        fragColor = vec4(camera_ray.direction * 0.5 + 0.5, 1.0);
    } else {
        vec3 color = hit.normal * 0.5 + 0.5;
        fragColor = vec4(color, 1.0);
    }
}
#endif
