/* Tetration C ABI — keep in sync with src/ffi/mod.rs (feature tetration-ffi). */
#ifndef TETRATION_H
#define TETRATION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bump when C symbols or calling conventions break (independent of crate semver). */
#define TET_ABI_VERSION 1u

/** Opaque read-only `.tet` handle. */
typedef struct TetHandle TetHandle;

uint32_t tet_abi_version(void);

TetHandle *tet_open(const char *path);
void tet_close(TetHandle *handle);

/** Valid until the next FFI call on this thread; do not free. */
const char *tet_last_error(void);
void tet_clear_error(void);

/** Caller must free with tet_string_free. */
char *tet_summary_json(TetHandle *handle);
char *tet_query_json(TetHandle *handle, const char *query_json);
char *tet_verify_json(const char *path);

void tet_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* TETRATION_H */
