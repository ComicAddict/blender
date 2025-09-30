/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "DNA_mesh_types.h"

#include "BKE_mesh.hh"

#include "BKE_mesh_types.hh"

#include "BKE_editmesh.hh"

#include "BLI_array_utils.hh"

#include "node_geometry_util.hh"

namespace blender::nodes::node_geo_mesh_topology_radial_offset_corner_cc {

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Int>("Corner Index")
      .implicit_field(NODE_DEFAULT_INPUT_INDEX_FIELD)
      .description("The edge to retrieve data from. Defaults to the edge from the context")
      .structure_type(StructureType::Field);
  b.add_input<decl::Int>("Offset")
      .min(0)
      .supports_field()
      .description("Which of the sorted corners to output");
  b.add_output<decl::Int>("Corner Index")
      .field_source_reference_all()
      .description(
          "A corner of the input edge in its face's winding order, chosen by the sort index");
  b.add_output<decl::Int>("Total").field_source().reference_pass({0}).description(
      "The number of faces or corners connected to each edge");
}

class RadialOffsetCorner final : public bke::MeshFieldInput {
  const Field<int> corner_index_;
  const Field<int> offset_;

 public:
  RadialOffsetCorner(Field<int> corner_index, Field<int> offset)
      : bke::MeshFieldInput(CPPType::get<int>(), "Radial Offset Corner of Edge"),
        corner_index_(std::move(corner_index)),
        offset_(std::move(offset))
  {
    category_ = Category::Generated;
  }

  GVArray get_varray_for_context(const Mesh &mesh,
                                 const AttrDomain domain,
                                 const IndexMask &mask) const final
  {
    const IndexRange corner_range(mesh.corners_num);

    const BMeshCreateParams bm_create_params = {false};
    BMeshFromMeshParams bm_convert_params{};
    bm_convert_params.calc_face_normal = true;
    bm_convert_params.calc_vert_normal = true;

    BMesh *bm = BKE_mesh_to_bmesh_ex(&mesh, &bm_create_params, &bm_convert_params);
    // mesh.
    // BMesh* bm = mesh.runtime->edit_mesh.get()->bm;
    
    const OffsetIndices faces = mesh.faces();

    BMFace *f;
    BMIter iter;
    BMLoop *l;
    int i = 0;

    Array<int> offset_corners(mask.min_array_size());
    BM_ITER_MESH (f, &iter, bm, BM_FACES_OF_MESH) {
    if (BM_elem_flag_test(f, BM_ELEM_TAG)) {
      BMLoop *l_iter, *l_first;
      l_iter = l_first = BM_FACE_FIRST_LOOP(f);
      do {
        offset_corners[i] = i;
        i++;
      } while ((l_iter = l_iter->next) != l_first);
    }
  }
  
  // mask.foreach_index_optimized<int>(GrainSize(2048), [&](const int selection_i) {
  //     const int corner = corner_indices[selection_i];
  //     const int offset = offsets[selection_i];
  //     if (!corner_to_face.index_range().contains(corner)) {
  //       offset_corners[selection_i] = 0;
  //       return;
  //     }
  //     const IndexRange face = faces[corner_to_face[corner]];
  //     const int corner_index_in_face = corner - face.start();
  //     offset_corners[selection_i] = face.start() + math::mod_periodic<int>(
  //                                                      corner_index_in_face + offset, face.size());
  //   });

    // const bke::MeshFieldContext context{mesh, domain};
    // fn::FieldEvaluator evaluator{context, &mask};
    // evaluator.add(corner_index_);
    // evaluator.add(offset_);
    // evaluator.evaluate();
    // const VArray<int> corner_indices = evaluator.get_evaluated<int>(0);
    // const VArray<int> offsets = evaluator.get_evaluated<int>(1);

    // const Span<int> corner_to_face = mesh.corner_to_face_map();

    // Array<int> offset_corners(mask.min_array_size());
    // mask.foreach_index_optimized<int>(GrainSize(2048), [&](const int selection_i) {
    //   const int corner = corner_indices[selection_i];
    //   const int offset = offsets[selection_i];
    //   if (!corner_to_face.index_range().contains(corner)) {
    //     offset_corners[selection_i] = 0;
    //     return;
    //   }
    //   const IndexRange face = faces[corner_to_face[corner]];
    //   const int corner_index_in_face = corner - face.start();
    //   offset_corners[selection_i] = face.start() + math::mod_periodic<int>(
    //                                                    corner_index_in_face + offset, face.size());
    // });
    BM_mesh_free(bm);

    return VArray<int>::from_container(std::move(offset_corners));
  }

  void for_each_field_input_recursive(FunctionRef<void(const FieldInput &)> fn) const override
  {
    corner_index_.node().for_each_field_input_recursive(fn);
    offset_.node().for_each_field_input_recursive(fn);
  }

  std::optional<AttrDomain> preferred_domain(const Mesh & /*mesh*/) const final
  {
    return AttrDomain::Edge;
  }
};

// class RadialOffsetCornersCountInput final : public bke::MeshFieldInput {
//  public:
//   RadialOffsetCornersCountInput() : bke::MeshFieldInput(CPPType::get<int>(), "Radal Face Corner Count")
//   {
//     category_ = Category::Generated;
//   }

//   GVArray get_varray_for_context(const Mesh &mesh,
//                                  const AttrDomain domain,
//                                  const IndexMask & /*mask*/) const final
//   {
//     if (domain != AttrDomain::Corner) {
//       return {};
//     }
//     Array<int> counts(mesh.edges_num, 0);
//     array_utils::count_indices(mesh.corner_edges(), counts);
//     return VArray<int>::from_container(std::move(counts));
//   }

//   uint64_t hash() const final
//   {
//     return 2345897985577;
//   }

//   bool is_equal_to(const fn::FieldNode &other) const final
//   {
//     return dynamic_cast<const RadialOffsetCornersCountInput *>(&other) != nullptr;
//   }

//   std::optional<AttrDomain> preferred_domain(const Mesh & /*mesh*/) const final
//   {
//     return AttrDomain::Corner;
//   }
// };

static void node_geo_exec(GeoNodeExecParams params)
{
  const Field<int> corner_index = params.extract_input<Field<int>>("Corner Index");
  // if (params.output_is_required("Total")) {
  //   params.set_output("Total",
  //                     Field<int>(std::make_shared<bke::EvaluateAtIndexInput>(
  //                         edge_index,
  //                         Field<int>(std::make_shared<RadialOffsetCornersCountInput>()),
  //                         AttrDomain::Edge)));
  // }
  // if (params.output_is_required("Corner Index")) {
  params.set_output("Corner Index",
                    Field<int>(std::make_shared<RadialOffsetCorner>(
                        corner_index,
                        params.extract_input<Field<int>>("Offset"))));
  // }
}

static void node_register()
{
  static blender::bke::bNodeType ntype;
  geo_node_type_base(&ntype, "GeometryNodeRadialOffsetCorners", GEO_NODE_MESH_TOPOLOGY_RADIAL_OFFSET_CORNERS);
  ntype.ui_name = "Radial Offset Corners";
  ntype.ui_description = "Retrieve radial offset face corners connected to edges";
  // ntype.enum_name_legacy = "RADIAL_OFFSET_CORNERS";
  ntype.nclass = NODE_CLASS_INPUT;
  ntype.geometry_node_execute = node_geo_exec;
  ntype.declare = node_declare;
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_geo_mesh_topology_corners_of_edge_cc
