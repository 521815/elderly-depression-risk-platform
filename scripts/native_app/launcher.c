#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    const char *home = getenv("HOME");
    if (home == NULL || home[0] == '\0') {
        home = "/Users/yellowich";
    }

    char script[4096];
    int written = snprintf(
        script, sizeof(script),
        "%s/Desktop/\xe5\xa4\xa7\xe5\x88\x9b/"
        "\xe8\x80\x81\xe5\xb9\xb4\xe6\x8a\x91\xe9\x83\x81\xe9\xa3\x8e\xe9\x99\xa9"
        "\xe9\xa2\x84\xe6\xb5\x8b\xe5\xa4\xa7\xe6\xa8\xa1\xe5\x9e\x8b/"
        "scripts/desktop_app.py",
        home
    );
    if (written < 0 || (size_t)written >= sizeof(script)) {
        fprintf(stderr, "project path too long\n");
        return 1;
    }

    if (access(script, R_OK) != 0) {
        fprintf(stderr, "desktop_app.py not found: %s\n", script);
        return 1;
    }

    setenv("PATH", "/opt/anaconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin", 1);
    setenv("LANG", "zh_CN.UTF-8", 1);

    char *const argv[] = {"/opt/anaconda3/bin/python", script, NULL};
    execv(argv[0], argv);
    fprintf(stderr, "failed to exec python: %s\n", strerror(errno));
    return 1;
}
