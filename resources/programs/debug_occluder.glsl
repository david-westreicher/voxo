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
#line 19

in vec2 uv;

uniform vec3 uCameraPos;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform usampler3D occluder_texture;
uniform ivec3 size;

Box bbox = Box(vec3(0.0), vec3(size));
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

out vec4 fragColor;

void main() {
    Ray camera_ray = compute_camera_ray(uInvProjection, uInvView, uCameraPos, 0, 0.0);
    Hit hit = dda(camera_ray, MAX_STEPS, occluder_texture, bbox);
    if (!hit.hit) {
        fragColor = vec4(camera_ray.direction * 0.5 + 0.5, 1.0); //vec4(0.0);
    } else {
        vec3 color = hit.normal * 0.5 + 0.5;
        fragColor = vec4(color, 1.0);
    }
}
#endif
