#include "BLI_array_utils.hh"
#include "BLI_vector_set.hh"

#include "DNA_mesh_types.h"
#include "DNA_meshdata_types.h"

#include "BKE_attribute_math.hh"
#include "BKE_mesh.hh"

#include "UI_interface.hh"
#include "UI_resources.hh"

#include "node_geometry_util.hh"

#include "voro++.hh"

namespace blender::nodes::node_geo_voronoi {

namespace {
struct AttributeOutputs {
  std::optional<std::string> cell_id;
};
}  // namespace

static void node_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Geometry>("Sites");
  b.add_input<decl::Geometry>("Domain");
  b.add_input<decl::Vector>("Min");
  b.add_input<decl::Vector>("Max");
  b.add_input<decl::Bool>("Periodic X");
  b.add_input<decl::Bool>("Periodic Y");
  b.add_input<decl::Bool>("Periodic Z");
  b.add_input<decl::Bool>("Boundary");
  b.add_input<decl::Int>("Group ID").implicit_field(implicit_field_inputs::index);
  b.add_output<decl::Geometry>("Voronoi");
  b.add_output<decl::Int>("Cell ID").field_on_all();
}

static Mesh *compute_voronoi(GeometrySet& sites, const GeometrySet& domain, AttributeOutputs& attribute_outputs, 
                            const float3 &min, const float3 &max, const Field<int>& group_id, 
                            bool x_p, bool y_p, bool z_p, bool boundary){
  const MeshComponent *mesh_comp = sites.get_component<MeshComponent>();
  const Mesh *site_mesh = mesh_comp->get();
  const Span<float3> positions = site_mesh->vert_positions();

  const bke::AttrDomain att_domain = bke::AttrDomain::Point;
  const int domain_size = site_mesh->attributes().domain_size(att_domain);
  bke::MeshFieldContext field_context{*site_mesh, att_domain};
  FieldEvaluator field_evaluator{field_context, domain_size};
  field_evaluator.add(group_id);
  field_evaluator.evaluate();
  const VArray<int> group_ids = field_evaluator.get_evaluated<int>(0);
  
  // Set the computational grid size
  const int n_x=7,n_y=7,n_z=14;

  // Create a container with the geometry given above, and make it
	// non-periodic in each of the three coordinates. Allocate space for
	// eight particles within each computational block.
	voro::container con(min[0],max[0],min[1],max[1],min[2],max[2],n_x,n_y,n_z,
			x_p,y_p,z_p,8);
  int i,j,k,l,n,id;

  i = 0;
  for (float3 p : positions){
    con.put(group_ids[i], p[0], p[1], p[2]);
    i++;
  }


  voro::voronoicell_neighbor c;
  voro::c_loop_all vl(con);
  
  std::vector<int> neigh, f_vert;
  std::vector<double> v;

  Vector<float3> verts;
  Vector<int> face_sizes;
  Vector<int> corner_verts;
  Vector<int> ids;
  double x,y,z;

  int offset = 0;

  if(vl.start()) do if(con.compute_cell(c,vl)){
    vl.pos(x,y,z);
    id=vl.pid();
    
    c.neighbors(neigh);
    c.face_vertices(f_vert);
    c.vertices(x,y,z,v);

    for(i = 0, j=0; i < neigh.size(); i++){
      if(neigh[i] != id && (boundary || neigh[i] > -1)){
        l = f_vert[j];
        n = f_vert[j];
        face_sizes.append(n);
        for(k=0; k < n; k++){
          l=3*f_vert[j+k+1];
          verts.append(float3(v[l],v[l+1],v[l+2]));
          corner_verts.append(offset);
          ids.append(id);
          offset++;
        }
      }
      j += f_vert[j]+1;
    }
  } while(vl.inc());
 

  Mesh *mesh = BKE_mesh_new_nomain(verts.size(), 0,face_sizes.size(), corner_verts.size());
  mesh->vert_positions_for_write().copy_from(verts);

  MutableSpan<int> face_offs = mesh->face_offsets_for_write();
  MutableSpan<int> corns = mesh->corner_verts_for_write();
  
  offset = 0;
  for(i = 0; i < face_sizes.size(); i++){
    int size = face_sizes[i];
    face_offs[i] = offset;
    for(j = 0; j < size; j++){
      corns[offset+j] = corner_verts[offset+j];
    }
    offset += size;
  }

  MutableAttributeAccessor mesh_attributes = mesh->attributes_for_write();
  SpanAttributeWriter<int> cell_id;

  if (attribute_outputs.cell_id) {
    cell_id = mesh_attributes.lookup_or_add_for_write_only_span<int>(
        *attribute_outputs.cell_id, AttrDomain::Point);
    std::copy(ids.begin(), ids.end(), cell_id.span.begin());
    cell_id.finish();
  }

  bke::mesh_calc_edges(*mesh, true, false);
  return mesh;
}

static void node_geo_exec(GeoNodeExecParams params)
{
  GeometrySet site_geometry = params.extract_input<GeometrySet>("Sites");
  const float3 min = params.extract_input<float3>("Min");
  const float3 max = params.extract_input<float3>("Max");
  const bool x_p = params.extract_input<bool>("Periodic X");
  const bool y_p = params.extract_input<bool>("Periodic Y");
  const bool z_p = params.extract_input<bool>("Periodic Z");
  const bool boundary = params.extract_input<bool>("Boundary");
  const GeometrySet domain = params.extract_input<GeometrySet>("Domain");
  Field<int> id_field = params.extract_input<Field<int>>("Group ID");

  // Map<int, std::unique_ptr<GeometrySet>> geometry_by_group_id;

  AttributeOutputs attribute_outputs;
  attribute_outputs.cell_id = params.get_output_anonymous_attribute_id_if_needed("Cell ID");

  if(site_geometry.has_mesh()){
    Mesh *voronoi = compute_voronoi(site_geometry, domain, attribute_outputs, 
                                    min, max, id_field, 
                                    x_p, y_p, z_p, boundary);
    site_geometry.replace_mesh(voronoi);
  } else {
    params.error_message_add(NodeWarningType::Error,
                           TIP_("Input should contain a mesh to compute Voronoi"));
  }

  params.set_output("Voronoi", std::move(site_geometry));
}

static void node_register()
{
  static blender::bke::bNodeType ntype;

  geo_node_type_base(&ntype, GEO_NODE_VORONOI, "Voronoi", NODE_CLASS_GEOMETRY);
  ntype.declare = node_declare;
  ntype.geometry_node_execute = node_geo_exec;
  blender::bke::node_register_type(&ntype);
}
NOD_REGISTER_NODE(node_register)

}