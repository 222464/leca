#pragma once

#include <vector>
#include <array>
#include <random>
#include <omp.h>

inline float sigmoid(
    float x
) {
    if (x < 0.0f) {
        float z = std::exp(x);

        return z / (1.0f + z);
    }
    
    return 1.0f / (1.0f + std::exp(-x));
}

class Leca {
public:
    struct Cell {
        std::array<float, 8> values;
        std::array<float, 16> actions;
        std::array<float, 16> valueTraces;
        std::array<float, 16> actionTraces;

        bool on;
        float valuePrev;

        Cell()
        :
        on(false),
        valuePrev(0.0f)
        {}
    };

private:
    std::mt19937 rng;

    int width;
    int height;

    std::vector<Cell> cells;

public:
    float vlr;
    float alr;
    float discount;
    float traceDecay;

    Leca()
    :
    vlr(0.01f),
    alr(0.1f),
    discount(0.99f),
    traceDecay(0.97f)
    {}

    void init(
        int width,
        int height,
        unsigned int seed
    );

    void step(
        const std::vector<bool> &inputs,
        float reward,
        bool learnEnabled
    );

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    bool getOn(
        int x,
        int y
    ) const {
        return cells[y + x * height].on;
    }

    bool getLastOn(
        int x
    ) const {
        return cells[height - 1 + x * height].on;
    }
};
