float halton(int base, int index) {
    float result = 0.;
    float f = 1.;
    while (index > 0)
    {
        f = f / float(base);
        result += f * float(index % base);
        index = index / base;
    }
    return result;
}

vec2 halton2D(int frame_counter) {
    frame_counter = frame_counter % 32;
    return vec2(halton(2, frame_counter), halton(3, frame_counter));
}

vec3 halton3D(int frame_counter) {
    frame_counter = frame_counter % 32;
    return vec3(halton(2, frame_counter), halton(3, frame_counter), halton(5, frame_counter));
}
