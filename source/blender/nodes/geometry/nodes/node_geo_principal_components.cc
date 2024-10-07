/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "BKE_attribute_math.hh"

#include "BLI_array.hh"
#include "BLI_array_utils.hh"
#include "BLI_generic_virtual_array.hh"
#include "BLI_math_quaternion.hh"
#include "BLI_math_solvers.h"
#include "BLI_virtual_array.hh"

#include "NOD_rna_define.hh"
#include "NOD_socket_search_link.hh"

#include "RNA_enum_types.hh"

#include "node_geometry_util.hh"

#include "UI_interface.hh"
#include "UI_resources.hh"

/**
 * Computes the principal components of a 3D vector set.
 *
 * Typically this is used to find the orientation of a point cloud where positions have maximum
 * variance along the first axis. The components output contains the variance of the input vectors
 * when rotated by the rotation output.
 *
 * Uses the Eigen SVD implementation for 3x3 matrices.
 *
 * https://en.wikipedia.org/wiki/Principal_component_analysis
 */

namespace blender::nodes::node_geo_principal_components_cc {

using math::Quaternion;

static void node_declare(NodeDeclarationBuilder &b)
{
  const bNode *node = b.node_or_null();

  b.add_input<decl::Geometry>("Geometry");
  b.add_input<decl::Bool>("Selection").default_value(true).field_on_all().hide_value();

  if (node != nullptr) {
    b.add_input<decl::Vector>("Position")
        .hide_value()
        .implicit_field_on_all(implicit_field_inputs::position);

    b.add_output<decl::Vector>("Mean");
    b.add_output<decl::Rotation>("Rotation");
    b.add_output<decl::Vector>("Components");
  }
}

static void node_layout(uiLayout *layout, bContext * /*C*/, PointerRNA *ptr)
{
  uiItemR(layout, ptr, "domain", UI_ITEM_NONE, "", ICON_NONE);
}

static void node_init(bNodeTree * /*tree*/, bNode *node)
{
  node->custom1 = int16_t(AttrDomain::Point);
}

static void node_gather_link_searches(GatherLinkSearchOpParams &params)
{
  const blender::bke::bNodeType &node_type = params.node_type();
  const NodeDeclaration &declaration = *params.node_type().static_declaration;
  search_link_ops_for_declarations(params, declaration.inputs);

  if (params.in_out() == SOCK_IN) {
    params.add_item(IFACE_("Position"), [node_type](LinkSearchOpParams &params) {
      bNode &node = params.add_node(node_type);
      params.update_and_connect_available_socket(node, "Position");
    });
  }
  else {
    for (const StringRefNull name : {"Mean", "Rotation", "Components"}) {
      params.add_item(IFACE_(name.c_str()), [node_type, name](LinkSearchOpParams &params) {
        bNode &node = params.add_node(node_type);
        params.update_and_connect_available_socket(node, name);
      });
    }
  }
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet geometry_set = params.get_input<GeometrySet>("Geometry");
  const bNode &node = params.node();
  const AttrDomain domain = AttrDomain(node.custom1);
  Vector<const GeometryComponent *> components = geometry_set.get_components();

  const Field<bool> selection_field = params.get_input<Field<bool>>("Selection");

  const Field<float3> input_field = params.get_input<Field<float3>>("Position");
  Vector<float3> data;
  for (const GeometryComponent *component : components) {
    const std::optional<AttributeAccessor> attributes = component->attributes();
    if (!attributes.has_value()) {
      continue;
    }
    if (attributes->domain_supported(domain)) {
      const bke::GeometryFieldContext field_context{*component, domain};
      fn::FieldEvaluator data_evaluator{field_context, attributes->domain_size(domain)};
      data_evaluator.add(input_field);
      data_evaluator.set_selection(selection_field);
      data_evaluator.evaluate();
      const VArray<float3> component_data = data_evaluator.get_evaluated<float3>(0);
      const IndexMask selection = data_evaluator.get_evaluated_selection_as_mask();

      const int next_data_index = data.size();
      data.resize(data.size() + selection.size());
      MutableSpan<float3> selected_data = data.as_mutable_span().slice(next_data_index,
                                                                       selection.size());
      array_utils::gather(component_data, selection, selected_data);
    }
  }

  float3 mean{0};
  Quaternion rotation{Quaternion::identity()};
  float3 principal_components{0};

  if (data.size() != 0) {
    float3 sum = std::accumulate(data.begin(), data.end(), float3(0.0f, 0.0f, 0.0f));
    mean = sum / data.size();

#if 0 /* Simple linear implementation. */
    float3 covar_diag = {0.0f, 0.0f, 0.0f};
    float3 covar_offdiag = {0.0f, 0.0f, 0.0f};
    for (const float3 &v : data) {
      const float3 dv = v - mean;
      covar_diag += {dv.x * dv.x, dv.y * dv.y, dv.z * dv.z};
      covar_offdiag += {dv.x * dv.y, dv.y * dv.z, dv.z * dv.x};
    }
    if (data.size() > 1) {
      covar_diag /= data.size() - 1;
      covar_offdiag /= data.size() - 1;
    }
    const float3x3 covar_matrix(float3(covar_diag.x, covar_offdiag.x, covar_offdiag.z),
                                float3(covar_offdiag.x, covar_diag.y, covar_offdiag.y),
                                float3(covar_offdiag.z, covar_offdiag.y, covar_diag.z));
#else /* Parallel-reduce implementation. */
    struct CovarianceValues {
      float3 diag = {0.0f, 0.0f, 0.0f};
      float3 offdiag = {0.0f, 0.0f, 0.0f};

      CovarianceValues &operator=(const CovarianceValues &other)
      {
        diag = other.diag;
        offdiag = other.offdiag;
        return *this;
      }

      CovarianceValues operator+(const CovarianceValues &other) const
      {
        return CovarianceValues{diag + other.diag, offdiag + other.offdiag};
      }

      float3x3 matrix() const
      {
        return float3x3(float3(diag.x, offdiag.x, offdiag.z),
                        float3(offdiag.x, diag.y, offdiag.y),
                        float3(offdiag.z, offdiag.y, diag.z));
      }
    };

    const CovarianceValues covar_values = threading::parallel_reduce(
        data.index_range(),
        4096,
        CovarianceValues(),
        [data, mean](const IndexRange range, CovarianceValues values) {
          for (const int64_t i : range) {
            const float3 dv = data[i] - mean;
            values = values + CovarianceValues{float3{dv.x * dv.x, dv.y * dv.y, dv.z * dv.z},
                                               float3{dv.x * dv.y, dv.y * dv.z, dv.z * dv.x}};
          }
          return values;
        },
        [](const CovarianceValues &a, const CovarianceValues &b) { return a + b; });
    const float3x3 covar_matrix = covar_values.matrix() *
                                  (1.0f / std::max(float(data.size() - 1), 1.0f));
#endif

    float3x3 U, V;
    float3 S;
    BLI_svd_m3(covar_matrix.ptr(), U.ptr(), S, V.ptr());

    rotation = math::to_quaternion(U);
    principal_components = S;
  }

  params.set_output("Mean", mean);
  params.set_output("Rotation", rotation);
  params.set_output("Components", principal_components);
}

static void node_rna(StructRNA *srna)
{
  RNA_def_node_enum(srna,
                    "domain",
                    "Domain",
                    "Which domain to read the data from",
                    rna_enum_attribute_domain_items,
                    NOD_inline_enum_accessors(custom1),
                    int(AttrDomain::Point));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(
      &ntype, GEO_NODE_PRINCIPAL_COMPONENTS, "Principal Components", NODE_CLASS_ATTRIBUTE);
  ntype.geometry_node_execute = node_geo_exec;
  ntype.initfunc = node_init;
  ntype.draw_buttons = node_layout;
  ntype.declare = node_declare;
  ntype.gather_link_search_ops = node_gather_link_searches;
  blender::bke::nodeRegisterType(&ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_principal_components_cc
