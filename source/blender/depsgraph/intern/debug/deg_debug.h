/* SPDX-FileCopyrightText: 2013 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup depsgraph
 */

#pragma once

#include <string>

#include "BKE_global.hh"  // IWYU pragma: keep

namespace blender::deg {

class DepsgraphDebug {
 public:
  DepsgraphDebug();

  bool do_time_debug() const;

  void begin_graph_evaluation();
  void end_graph_evaluation();

  /* NOTE: Corresponds to G_DEBUG_DEPSGRAPH_* flags. */
  int flags;

  /* Name of this dependency graph (is used for debug prints, helping to distinguish graphs
   * created for different view layer). */
  std::string name;

 protected:
  /* Maximum number of counters used to calculate frame rate of depsgraph update. */
  static const constexpr int MAX_FPS_COUNTERS = 64;

  /* Point in time when last graph evaluation began.
   * Is initialized from begin_graph_evaluation() when time debug is enabled.
   */
  double graph_evaluation_start_time_;
};

#define DEG_DEBUG_PRINTF(depsgraph, type, ...) \
  do { \
    if (DEG_debug_flags_get(depsgraph) & G_DEBUG_DEPSGRAPH_##type) { \
      DEG_debug_print_begin(depsgraph); \
      fprintf(stdout, __VA_ARGS__); \
    } \
  } while (0)

#define DEG_GLOBAL_DEBUG_PRINTF(type, ...) \
  do { \
    if (G.debug & G_DEBUG_DEPSGRAPH_##type) { \
      fprintf(stdout, __VA_ARGS__); \
    } \
  } while (0)

#define DEG_ERROR_PRINTF(...) \
  do { \
    fprintf(stderr, __VA_ARGS__); \
    fflush(stderr); \
  } while (0)

bool terminal_do_color();
std::string color_for_pointer(const void *pointer);
std::string color_end();

}  // namespace blender::deg
