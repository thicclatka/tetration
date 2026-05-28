/* Minimal C consumer for libtetration (Phase 11).
 *
 * Build (from repo root, after `cargo build --release --no-default-features --features tetration-ffi`):
 *
 *   cc -std=c11 -Wall -Wextra -I include examples/ffi_query.c \
 *     -L target/release -ltetration -o target/release/ffi_query
 *
 * Run (macOS):
 *   DYLD_LIBRARY_PATH=target/release target/release/ffi_query fixtures/small/tet/sample.tet
 *
 * Run (Linux):
 *   LD_LIBRARY_PATH=target/release target/release/ffi_query fixtures/small/tet/sample.tet
 */

#include "tetration.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path;
    const char *query =
        "{\"dataset\":\"temperature\",\"mean\":[]}";

    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.tet> [query-json]\n", argv[0]);
        return 2;
    }
    path = argv[1];
    if (argc >= 3) {
        query = argv[2];
    }

    if (tet_abi_version() != TET_ABI_VERSION) {
        fprintf(stderr, "ABI mismatch: header=%u library=%u\n", (unsigned)TET_ABI_VERSION,
                (unsigned)tet_abi_version());
        return 1;
    }

    TetHandle *handle = tet_open(path);
    if (handle == NULL) {
        fprintf(stderr, "tet_open: %s\n", tet_last_error());
        return 1;
    }

    char *out = tet_query_json(handle, query);
    if (out == NULL) {
        fprintf(stderr, "tet_query_json: %s\n", tet_last_error());
        tet_close(handle);
        return 1;
    }

    /* Print the operation_mean field if present, else the full JSON line. */
    const char *key = "\"operation_mean\":";
    const char *hit = strstr(out, key);
    if (hit != NULL) {
        fputs(hit, stdout);
    } else {
        fputs(out, stdout);
    }
    fputc('\n', stdout);

    tet_string_free(out);
    tet_close(handle);
    return 0;
}
