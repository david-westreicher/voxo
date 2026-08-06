#version 420

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#line 18

in vec2 uv;

layout(binding = 0) uniform sampler2D u_albedo;
layout(binding = 1) uniform sampler2D u_irradiance;
layout(binding = 2) uniform sampler2D u_material;
layout(binding = 3) uniform sampler2D u_depth;

layout(location = 0) out vec3 fragColor;

void main() {
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        return;
    }
    vec3 albedo = texture(u_albedo, uv).rgb;
    vec3 irradiance = texture(u_irradiance, uv).rgb;
    float emissive = max(0, texture(u_material, uv).a);

    fragColor = albedo * (irradiance + emissive * 10.0);
}
#endif
