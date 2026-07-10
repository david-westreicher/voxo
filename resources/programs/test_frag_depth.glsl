#version 400

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 vUV;

void main() {
    gl_Position = vec4(in_position, 1);
    vUV = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

layout(location = 0) out vec3 u_albedo;
layout(location = 1) out vec3 u_normal;

void main() {
    u_albedo = vec3(1.0, 0.0, 0.0);
    u_normal = vec3(0.0, 1.0, 0.0);
    gl_FragDepth = 0.1;
}

#endif
