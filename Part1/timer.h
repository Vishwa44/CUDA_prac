#include <time.h>
#include <stdio.h>  // Added this for printf

typedef struct {
    clock_t start;
    clock_t stop;
} Timer;

void startTime(Timer* timer) {
    timer->start = clock();
}

void stopTime(Timer* timer) {
    timer->stop = clock();
}

void printElapsedTime(Timer timer, const char* desc, const char* color) {
    double elapsed = ((double)(timer.stop - timer.start)) / CLOCKS_PER_SEC;
    printf("%s: %f seconds\n", desc, elapsed);
}