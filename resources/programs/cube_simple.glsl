#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform vec3 scale;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position * scale, 1.0);
    gl_Position = m_proj * p;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform vec3 color;

void main() {
    fragColor = vec4(color, 1.0);
}
#endif
