use naga::{ArraySize, GlobalVariable, Type, TypeInner, VectorSize};

#[derive(Debug, Default, Clone)]
pub struct Bindings {
    module: naga::Module,
    bindings: Vec<Binding>,
}

impl Bindings {
    pub fn new(module: naga::Module) -> Self {
        let bindings = bindings(&module).collect();
        Self { module, bindings }
    }

    pub fn find_name(&self, name: &str) -> Option<Binding> {
        self.bindings().find(|binding| binding.name() == Some(name))
    }

    pub fn has_name(&self, name: &str) -> bool {
        self.find_name(name).is_some()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.iter().filter_map(|binding| binding.name())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Binding> {
        self.bindings.iter()
    }

    pub fn get(&self, index: usize) -> Option<&Binding> {
        self.bindings.get(index)
    }

    pub fn bindings(&self) -> impl Iterator<Item = Binding> + '_ {
        self.bindings.iter().cloned()
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
    pub variable: GlobalVariable,
    pub ty: Type,
}

impl Binding {
    pub fn name(&self) -> Option<&str> {
        self.variable.name.as_ref().map(|s| s.as_str())
    }
}

pub fn bindings(
    module: &naga::Module,
) -> impl Iterator<Item = Binding> + '_ {
    module
        .global_variables
        .iter()
        .filter(|(_, var)| var.binding.is_some())
        .map(move |(_handle, variable)| {
            let binding = variable.binding.as_ref().unwrap();
            let ty = &module.types[variable.ty];
            Binding {
                group: binding.group,
                binding: binding.binding,
                variable: variable.clone(),
                ty: ty.clone(),
            }
        })
}

pub fn type_size(module: &naga::Module, ty: &naga::Type) -> u64 {
    match &ty.inner {
        TypeInner::Scalar(scalar) => scalar.width as u64,
        TypeInner::Vector { size, scalar } => scalar.width as u64 * vector_size(size),
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => scalar.width as u64 * vector_size(columns) * vector_size(rows),
        TypeInner::Atomic(scalar) => scalar.width as u64,
        TypeInner::Pointer { .. } => 8,
        TypeInner::ValuePointer { .. } => 8,
        TypeInner::Array { base, size, stride } => {
            let base_size = type_size(module, &module.types[*base]);
            let count = match size {
                ArraySize::Constant(count) => count.get() as u64,
                ArraySize::Pending(_) => 0,
                ArraySize::Dynamic => 0,
            };
            let element_size = std::cmp::max(base_size, *stride as u64);
            count * element_size
        }
        TypeInner::Struct { members, span } => {
            let _ = span;
            let mut offset = 0;
            for member in members {
                let member_size = type_size(module, &module.types[member.ty]);
                offset = align_to(offset, 16);
                offset += member_size;
            }
            align_to(offset, 16)
        }
        TypeInner::Image { .. } => 0,
        TypeInner::Sampler { .. } => 0,
        TypeInner::BindingArray { .. } => 0,
        TypeInner::AccelerationStructure { .. } => 8,
        TypeInner::RayQuery { .. } => 0,
    }
}

fn vector_size(size: &VectorSize) -> u64 {
    match size {
        VectorSize::Bi => 2,
        VectorSize::Tri => 3,
        VectorSize::Quad => 4,
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::{ArraySize, Handle, Module, ScalarKind, Type, TypeInner, VectorSize};
    use std::num::NonZeroU32;

    fn create_module() -> Module {
        Module::default()
    }

    fn add_type(module: &mut Module, inner: TypeInner) -> Handle<Type> {
        module
            .types
            .insert(Type { name: None, inner }, Default::default())
    }

    #[test]
    fn test_scalar_sizes() {
        let mut module = create_module();

        let i32_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            }),
        );
        assert_eq!(
            type_size(&module, &module.types[i32_type]),
            4
        );

        let f64_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Float,
                width: 8,
            }),
        );
        assert_eq!(
            type_size(&module, &module.types[f64_type]),
            8
        );

        let bool_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Bool,
                width: 1,
            }),
        );
        assert_eq!(
            type_size(&module, &module.types[bool_type]),
            1
        );
    }

    #[test]
    fn test_vector_sizes() {
        let mut module = create_module();

        let vec2_type = add_type(
            &mut module,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: naga::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                },
            },
        );
        assert_eq!(
            type_size(&module, &module.types[vec2_type]),
            8
        );

        let vec4_type = add_type(
            &mut module,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: naga::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
            },
        );
        assert_eq!(
            type_size(&module, &module.types[vec4_type]),
            16
        );
    }

    #[test]
    fn test_matrix_sizes() {
        let mut module = create_module();

        let mat3x3_type = add_type(
            &mut module,
            TypeInner::Matrix {
                columns: VectorSize::Tri,
                rows: VectorSize::Tri,
                scalar: naga::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                },
            },
        );
        assert_eq!(
            type_size(&module, &module.types[mat3x3_type]),
            36
        );
    }

    #[test]
    fn test_atomic_sizes() {
        let mut module = create_module();

        let atomic_i32_type = add_type(
            &mut module,
            TypeInner::Atomic(naga::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            }),
        );
        assert_eq!(
            type_size(&module, &module.types[atomic_i32_type]),
            4
        );
    }

    #[test]
    fn test_array_sizes() {
        let mut module = create_module();

        let f32_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            }),
        );
        let fixed_array_type = add_type(
            &mut module,
            TypeInner::Array {
                base: f32_type,
                size: ArraySize::Constant(NonZeroU32::new(10).unwrap()),
                stride: 4,
            },
        );
        assert_eq!(
            type_size(&module, &module.types[fixed_array_type]),
            40
        );

        let dynamic_array_type = add_type(
            &mut module,
            TypeInner::Array {
                base: f32_type,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        );
        assert_eq!(
            type_size(
                &module,
                &module.types[dynamic_array_type]
            ),
            0
        );
    }

    #[test]
    fn test_struct_sizes() {
        let mut module = create_module();

        let f32_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            }),
        );
        let vec3_type = add_type(
            &mut module,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: naga::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                },
            },
        );

        let struct_type = add_type(
            &mut module,
            TypeInner::Struct {
                members: vec![
                    naga::StructMember {
                        name: None,
                        ty: f32_type,
                        binding: None,
                        offset: 0,
                    },
                    naga::StructMember {
                        name: None,
                        ty: vec3_type,
                        binding: None,
                        offset: 16,
                    },
                ],
                span: 32,
            },
        );
        assert_eq!(
            type_size(&module, &module.types[struct_type]),
            32
        );
    }

    #[test]
    fn test_special_types() {
        let mut module = create_module();

        let image_type = add_type(
            &mut module,
            TypeInner::Image {
                dim: naga::ImageDimension::D2,
                arrayed: false,
                class: naga::ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        );
        assert_eq!(
            type_size(&module, &module.types[image_type]),
            0
        );

        let sampler_type = add_type(&mut module, TypeInner::Sampler { comparison: false });
        assert_eq!(
            type_size(&module, &module.types[sampler_type]),
            0
        );

        let accel_struct_type = add_type(&mut module, TypeInner::AccelerationStructure { vertex_return: false });
        assert_eq!(
            type_size(&module, &module.types[accel_struct_type]),
            8
        );
    }

    #[test]
    fn test_scalar_size_trait() {
        let i32_scalar = naga::Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        };
        assert_eq!(i32_scalar.width as u64, 4);

        let bool_scalar = naga::Scalar {
            kind: ScalarKind::Bool,
            width: 1,
        };
        assert_eq!(bool_scalar.width as u64, 1);
    }
}
