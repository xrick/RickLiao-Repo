#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

// Function to calculate the modified Bessel function of the first kind, order 0
double besselI0(double x) {
    double sum = 1.0;
    double u = 1.0;
    double half_x = x / 2.0;
    for (int k = 1; k <= 50; k++) {
        u *= (half_x * half_x) / (k * k);
        sum += u;
    }
    return sum;
}

// Function to apply Kaiser window to the signal
void apply_kaiser_window(double* signal, int length, double beta) {
    double denominator = besselI0(beta);
    for (int n = 0; n < length; n++) {
        double ratio = (2.0 * n) / (length - 1) - 1.0;
        double multiplier = besselI0(beta * sqrt(1 - ratio * ratio)) / denominator;
        signal[n] *= multiplier;
    }
}

// Function to read WAV file (simplified)
int read_wav_file(const char* filename, double** signal, int* length) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Could not open file");
        return -1;
    }

    // Skipping WAV header (simplified)
    fseek(file, 44, SEEK_SET);

    // Get the file size to determine the number of samples
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    *length = (file_size - 44) / sizeof(short);
    rewind(file);
    fseek(file, 44, SEEK_SET);

    // Allocate memory for signal and read the data
    *signal = (double*)malloc(*length * sizeof(double));
    short* buffer = (short*)malloc(*length * sizeof(short));
    fread(buffer, sizeof(short), *length, file);

    // Normalize the signal (convert to double)
    for (int i = 0; i < *length; i++) {
        (*signal)[i] = buffer[i] / 32768.0;
    }

    free(buffer);
    fclose(file);
    return 0;
}

// Function to write WAV file (simplified)
int write_wav_file(const char* filename, double* signal, int length) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Could not open file");
        return -1;
    }

    // Write a basic WAV header (simplified)
    short bits_per_sample = 16;
    short num_channels = 1;
    int sample_rate = 44100;
    int byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    short block_align = num_channels * bits_per_sample / 8;
    int data_size = length * num_channels * bits_per_sample / 8;

    fwrite("RIFF", 1, 4, file);
    int chunk_size = 36 + data_size;
    fwrite(&chunk_size, 4, 1, file);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    int subchunk1_size = 16;
    fwrite(&subchunk1_size, 4, 1, file);
    short audio_format = 1;
    fwrite(&audio_format, 2, 1, file);
    fwrite(&num_channels, 2, 1, file);
    fwrite(&sample_rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file);
    fwrite(&block_align, 2, 1, file);
    fwrite(&bits_per_sample, 2, 1, file);
    fwrite("data", 1, 4, file);
    fwrite(&data_size, 4, 1, file);

    // Convert the signal back to short and write to file
    short* buffer = (short*)malloc(length * sizeof(short));
    for (int i = 0; i < length; i++) {
        buffer[i] = (short)(signal[i] * 32767);
    }
    fwrite(buffer, sizeof(short), length, file);

    free(buffer);
    fclose(file);
    return 0;
}

int main() {
    const char* input_filename = "input.wav";
    const char* output_filename = "output.wav";
    double* signal;
    int length;

    // Read the WAV file
    if (read_wav_file(input_filename, &signal, &length) != 0) {
        return -1;
    }

    // Apply the Kaiser window with a beta value of 5.0
    double beta = 5.0;
    apply_kaiser_window(signal, length, beta);

    // Write the output WAV file
    if (write_wav_file(output_filename, signal, length) != 0) {
        free(signal);
        return -1;
    }

    free(signal);
    printf("Kaiser window applied successfully.\n");
    return 0;
}
