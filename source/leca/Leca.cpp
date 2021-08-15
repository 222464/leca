#include "Leca.h"

void Leca::init(
    int width,
    int height,
    unsigned int seed
) {
    this->width = width;
    this->height = height;

    rng.seed(seed);

    cells.resize(width * height);

    std::uniform_real_distribution<float> valueDist(-0.001f, 0.001f);

    for (int i = 0; i < cells.size(); i++) {
        for (int j = 0; j < cells[i].values.size(); j++) {
            cells[i].values[j] = 0.0f;
            cells[i].actions[j] = valueDist(rng);
            cells[i].valueTraces[j] = 0.0f;
            cells[i].actionTraces[j] = 0.0f;
        }
    }
}

void Leca::step(
    const std::vector<bool> &inputs,
    float reward,
    bool learnEnabled
) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int y = 0; y < height; y++) {
        #pragma omp parallel for
        for (int x = 0; x < width; x++) {
            int i = y + x * height;

            int cellIndex0 = 0;

            for (int dx = -1; dx <= 1; dx++) {
                int ix = x + dx;

                if (ix < 0 || ix >= width)
                    continue;

                if (y == 0) {
                    if (inputs[ix])
                        cellIndex0 = cellIndex0 | (1 << (dx + 1));
                }
                else {
                    if (cells[(y - 1) + ix * height].on)
                        cellIndex0 = cellIndex0 | (1 << (dx + 1));
                }
            }

            int cellIndex1 = cellIndex0 | (1 << 3);

            float prob = sigmoid(cells[i].actions[cellIndex1] - cells[i].actions[cellIndex0]);

            cells[i].on = dist01(rng) < prob;

            int cellIndex = cells[i].on ? cellIndex1 : cellIndex0;

            float value = cells[i].values[cellIndex0];

            float target = reward + discount * value;
            float tdError = target - cells[i].valuePrev;

            cells[i].valuePrev = value;

            // Traces
            for (int j = 0; j < cells[i].values.size(); j++) {
                if (learnEnabled)
                    cells[i].values[j] += vlr * tdError * cells[i].valueTraces[j];

                cells[i].valueTraces[j] *= traceDecay;
            }

            cells[i].valueTraces[cellIndex0] = 1.0f;

            for (int j = 0; j < cells[i].actions.size(); j++) {
                if (learnEnabled)
                    cells[i].actions[j] += alr * tdError * cells[i].actionTraces[j];

                cells[i].actionTraces[j] *= traceDecay;
            }

            if (cells[i].on) {
                cells[i].actionTraces[cellIndex0] = -1.0f;
                cells[i].actionTraces[cellIndex1] = 1.0f;
            }
            else {
                cells[i].actionTraces[cellIndex0] = 1.0f;
                cells[i].actionTraces[cellIndex1] = -1.0f;
            }
        }
    }
}
