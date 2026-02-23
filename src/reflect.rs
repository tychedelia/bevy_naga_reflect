use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, GlobalVariable, Handle, ImageClass, ImageDimension, ScalarKind,
    StorageAccess, StorageFormat, TypeInner, VectorSize,
};

use bevy::render::render_resource::binding_types::{
    sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_1d, texture_2d,
    texture_2d_array, texture_2d_array_multisampled, texture_2d_multisampled, texture_3d,
    texture_3d_multisampled, texture_cube, texture_cube_array, texture_cube_array_multisampled,
    texture_cube_multisampled, texture_storage_2d, uniform_buffer_sized,
};
use bevy::render::render_resource::{
    BindGroupLayoutEntry, BindGroupLayoutEntryBuilder, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

use std::num::NonZeroU64;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeKind {
    Scalar,
    Vector,
    Matrix,
    Array,
    Struct,
    Image,
    Sampler,
    Atomic,
    Pointer,
    ValuePointer,
    BindingArray,
    AccelerationStructure,
    RayQuery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    Sint,
    Uint,
    Float,
    AbstractInt,
    AbstractFloat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterCategory {
    Uniform,
    Storage { read_only: bool },
    Texture,
    StorageTexture,
    Sampler,
}

#[derive(Debug)]
pub(crate) struct CachedParameter {
    pub(crate) var_handle: Handle<GlobalVariable>,
    pub(crate) group: u32,
    pub(crate) binding: u32,
    pub(crate) type_handle: Handle<naga::Type>,
    pub(crate) visibility: ShaderStages,
    pub(crate) category: ParameterCategory,
}

pub struct ShaderReflection {
    pub(crate) module: naga::Module,
    pub(crate) module_info: ModuleInfo,
    pub(crate) parameters: Vec<CachedParameter>,
}

impl std::fmt::Debug for ShaderReflection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderReflection")
            .field("parameter_count", &self.parameters.len())
            .finish()
    }
}

impl Default for ShaderReflection {
    fn default() -> Self {
        Self::new(naga::Module::default()).expect("Default module should validate")
    }
}

impl Clone for ShaderReflection {
    fn clone(&self) -> Self {
        Self::new(self.module.clone()).expect("Already-validated module should re-validate")
    }
}

impl ShaderReflection {
    pub fn new(
        module: naga::Module,
    ) -> Result<Self, naga::WithSpan<naga::valid::ValidationError>> {
        let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
        let module_info = validator.validate(&module)?;
        let parameters = Self::build_parameters(&module, &module_info);

        Ok(Self {
            module,
            module_info,
            parameters,
        })
    }

    fn build_parameters(module: &naga::Module, module_info: &ModuleInfo) -> Vec<CachedParameter> {
        let mut parameters = Vec::new();

        for (handle, variable) in module.global_variables.iter() {
            let Some(ref resource_binding) = variable.binding else {
                continue;
            };


            let mut visibility = ShaderStages::NONE;
            for (ep_index, ep) in module.entry_points.iter().enumerate() {
                let ep_info = module_info.get_entry_point(ep_index);
                let usage = ep_info[handle];
                if !usage.is_empty() {
                    visibility |= match ep.stage {
                        naga::ShaderStage::Vertex => ShaderStages::VERTEX,
                        naga::ShaderStage::Fragment => ShaderStages::FRAGMENT,
                        naga::ShaderStage::Compute => ShaderStages::COMPUTE,
                        _ => ShaderStages::NONE,
                    };
                }
            }


            if visibility == ShaderStages::NONE {
                continue;
            }


            let ty = &module.types[variable.ty];
            let category = match &ty.inner {
                TypeInner::Image {
                    class: ImageClass::Storage { .. },
                    ..
                } => ParameterCategory::StorageTexture,
                TypeInner::Image { .. } => ParameterCategory::Texture,
                TypeInner::Sampler { .. } => ParameterCategory::Sampler,
                _ => match variable.space {
                    AddressSpace::Uniform => ParameterCategory::Uniform,
                    AddressSpace::Storage { access } => ParameterCategory::Storage {
                        read_only: access == StorageAccess::LOAD,
                    },
                    _ => ParameterCategory::Uniform,
                },
            };

            parameters.push(CachedParameter {
                var_handle: handle,
                group: resource_binding.group,
                binding: resource_binding.binding,
                type_handle: variable.ty,
                visibility,
                category,
            });
        }

        parameters
    }

    pub fn module(&self) -> &naga::Module {
        &self.module
    }

    pub fn parameters(&self) -> impl Iterator<Item = ParameterReflection<'_>> {
        self.parameters.iter().map(move |cached| ParameterReflection {
            module: &self.module,
            module_info: &self.module_info,
            cached,
        })
    }

    pub fn parameter(&self, name: &str) -> Option<ParameterReflection<'_>> {
        self.parameters().find(|p| p.name() == Some(name))
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn parameter_at(&self, index: usize) -> Option<ParameterReflection<'_>> {
        self.parameters.get(index).map(|cached| ParameterReflection {
            module: &self.module,
            module_info: &self.module_info,
            cached,
        })
    }

    pub fn has_parameter(&self, name: &str) -> bool {
        self.parameter(name).is_some()
    }

    pub fn entry_points(&self) -> impl Iterator<Item = EntryPointReflection<'_>> {
        self.module
            .entry_points
            .iter()
            .enumerate()
            .map(move |(i, ep)| EntryPointReflection {
                module_info: &self.module_info,
                entry_point: ep,
                index: i,
            })
    }

    pub fn bind_group_layout(&self, group: u32) -> Vec<BindGroupLayoutEntry> {
        self.parameters
            .iter()
            .filter(|p| p.group == group)
            .map(|cached| {
                let variable = &self.module.global_variables[cached.var_handle];
                let ty = &self.module.types[cached.type_handle];
                let builder = entry_builder(variable, &self.module, ty);
                builder.build(cached.binding, cached.visibility)
            })
            .collect()
    }

    pub fn create_bindings(
        &self,
        reflected: &dyn bevy::reflect::PartialReflect,
        render_device: &bevy::render::renderer::RenderDevice,
        gpu_images: &bevy::render::render_asset::RenderAssets<bevy::render::texture::GpuImage>,
    ) -> Vec<(u32, bevy::render::render_resource::OwnedBindingResource)> {
        let mut bindings = Vec::new();

        for cached in &self.parameters {
            let variable = &self.module.global_variables[cached.var_handle];
            let ty = &self.module.types[cached.type_handle];
            let Some(ref name) = variable.name else {
                continue;
            };

            let Some(field_value) = crate::binding::find_field(reflected, name) else {
                bevy::log::warn!("Field not found in reflected type: {:?}", name);
                continue;
            };

            let resource = crate::binding::generate_binding_resource(
                field_value,
                &self.module,
                ty,
                render_device,
                gpu_images,
            );
            bindings.push((cached.binding, resource));
        }

        bindings
    }

    pub fn diff(&self, other: &ShaderReflection) -> ShaderDiff {
        let old_names: Vec<String> = self
            .parameters()
            .filter_map(|p| p.name().map(|s| s.to_string()))
            .collect();
        let new_names: Vec<String> = other
            .parameters()
            .filter_map(|p| p.name().map(|s| s.to_string()))
            .collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut retained = Vec::new();
        let mut type_changed = Vec::new();

        for name in &new_names {
            if !old_names.contains(name) {
                added.push(name.clone());
            }
        }

        for name in &old_names {
            if !new_names.contains(name) {
                removed.push(name.clone());
            } else {
                retained.push(name.clone());
            }
        }

        for name in &retained {
            let old_param = self.parameter(name).unwrap();
            let new_param = other.parameter(name).unwrap();
            let old_kind = old_param.ty().kind();
            let new_kind = new_param.ty().kind();
            if old_kind != new_kind {
                type_changed.push((name.clone(), old_kind, new_kind));
            }
        }

        ShaderDiff {
            added,
            removed,
            retained,
            type_changed,
        }
    }
}

pub struct ParameterReflection<'a> {
    module: &'a naga::Module,
    #[allow(dead_code)]
    module_info: &'a ModuleInfo,
    cached: &'a CachedParameter,
}

impl ParameterReflection<'_> {
    pub fn name(&self) -> Option<&str> {
        self.module.global_variables[self.cached.var_handle]
            .name
            .as_deref()
    }

    pub fn group(&self) -> u32 {
        self.cached.group
    }

    pub fn binding(&self) -> u32 {
        self.cached.binding
    }

    pub fn category(&self) -> ParameterCategory {
        self.cached.category
    }

    pub fn visibility(&self) -> ShaderStages {
        self.cached.visibility
    }

    pub fn ty(&self) -> TypeReflection<'_> {
        TypeReflection {
            module: self.module,
            ty: &self.module.types[self.cached.type_handle],
        }
    }

    pub fn layout(&self) -> TypeLayout<'_> {
        TypeLayout {
            module: self.module,
            ty: &self.module.types[self.cached.type_handle],
        }
    }

    pub fn var_handle(&self) -> Handle<GlobalVariable> {
        self.cached.var_handle
    }
}


pub struct TypeReflection<'a> {
    module: &'a naga::Module,
    ty: &'a naga::Type,
}

impl<'a> TypeReflection<'a> {
    pub fn kind(&self) -> TypeKind {
        match &self.ty.inner {
            TypeInner::Scalar(_) => TypeKind::Scalar,
            TypeInner::Vector { .. } => TypeKind::Vector,
            TypeInner::Matrix { .. } => TypeKind::Matrix,
            TypeInner::Array { .. } => TypeKind::Array,
            TypeInner::Struct { .. } => TypeKind::Struct,
            TypeInner::Image { .. } => TypeKind::Image,
            TypeInner::Sampler { .. } => TypeKind::Sampler,
            TypeInner::Atomic(_) => TypeKind::Atomic,
            TypeInner::Pointer { .. } => TypeKind::Pointer,
            TypeInner::ValuePointer { .. } => TypeKind::ValuePointer,
            TypeInner::BindingArray { .. } => TypeKind::BindingArray,
            TypeInner::AccelerationStructure { .. } => TypeKind::AccelerationStructure,
            TypeInner::RayQuery { .. } => TypeKind::RayQuery,
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.ty.name.as_deref()
    }

    pub fn inner(&self) -> &naga::TypeInner {
        &self.ty.inner
    }

    pub fn scalar_type(&self) -> Option<ScalarType> {
        let scalar = match &self.ty.inner {
            TypeInner::Scalar(s) => Some(s),
            TypeInner::Vector { scalar, .. } => Some(scalar),
            TypeInner::Matrix { scalar, .. } => Some(scalar),
            TypeInner::Atomic(s) => Some(s),
            _ => None,
        }?;
        Some(match scalar.kind {
            ScalarKind::Bool => ScalarType::Bool,
            ScalarKind::Sint => ScalarType::Sint,
            ScalarKind::Uint => ScalarType::Uint,
            ScalarKind::Float => ScalarType::Float,
            ScalarKind::AbstractInt => ScalarType::AbstractInt,
            ScalarKind::AbstractFloat => ScalarType::AbstractFloat,
        })
    }

    pub fn scalar_width(&self) -> Option<u8> {
        match &self.ty.inner {
            TypeInner::Scalar(s) => Some(s.width),
            TypeInner::Vector { scalar, .. } => Some(scalar.width),
            TypeInner::Matrix { scalar, .. } => Some(scalar.width),
            TypeInner::Atomic(s) => Some(s.width),
            _ => None,
        }
    }

    pub fn vector_size(&self) -> Option<u32> {
        match &self.ty.inner {
            TypeInner::Vector { size, .. } => Some(vector_size_to_u32(size)),
            _ => None,
        }
    }

    pub fn columns(&self) -> Option<u32> {
        match &self.ty.inner {
            TypeInner::Matrix { columns, .. } => Some(vector_size_to_u32(columns)),
            _ => None,
        }
    }

    pub fn rows(&self) -> Option<u32> {
        match &self.ty.inner {
            TypeInner::Matrix { rows, .. } => Some(vector_size_to_u32(rows)),
            _ => None,
        }
    }

    pub fn element_type(&self) -> Option<TypeReflection<'a>> {
        match &self.ty.inner {
            TypeInner::Array { base, .. } => Some(TypeReflection {
                module: self.module,
                ty: &self.module.types[*base],
            }),
            _ => None,
        }
    }

    pub fn element_count(&self) -> Option<u32> {
        match &self.ty.inner {
            TypeInner::Array { size, .. } => match size {
                ArraySize::Constant(n) => Some(n.get()),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn fields(&self) -> Option<Vec<FieldReflection<'a>>> {
        match &self.ty.inner {
            TypeInner::Struct { members, .. } => Some(
                members
                    .iter()
                    .map(|member| FieldReflection {
                        module: self.module,
                        member,
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn field(&self, name: &str) -> Option<FieldReflection<'a>> {
        self.fields()?.into_iter().find(|f| f.name() == Some(name))
    }

    pub fn image_dimension(&self) -> Option<ImageDimension> {
        match &self.ty.inner {
            TypeInner::Image { dim, .. } => Some(*dim),
            _ => None,
        }
    }

    pub fn image_class(&self) -> Option<&ImageClass> {
        match &self.ty.inner {
            TypeInner::Image { class, .. } => Some(class),
            _ => None,
        }
    }

    pub fn is_comparison_sampler(&self) -> bool {
        matches!(&self.ty.inner, TypeInner::Sampler { comparison: true })
    }
}


pub struct TypeLayout<'a> {
    module: &'a naga::Module,
    ty: &'a naga::Type,
}

impl TypeLayout<'_> {
    pub fn size(&self) -> u64 {
        type_size(self.module, self.ty)
    }

    pub fn alignment(&self) -> u64 {
        match &self.ty.inner {
            TypeInner::Scalar(s) => s.width as u64,
            TypeInner::Vector { scalar, .. } => scalar.width as u64,
            TypeInner::Matrix { scalar, .. } => scalar.width as u64,
            TypeInner::Struct { .. } => 16,
            _ => 4,
        }
    }
}


pub struct FieldReflection<'a> {
    module: &'a naga::Module,
    member: &'a naga::StructMember,
}

impl<'a> FieldReflection<'a> {
    pub fn name(&self) -> Option<&str> {
        self.member.name.as_deref()
    }

    pub fn ty(&self) -> TypeReflection<'a> {
        TypeReflection {
            module: self.module,
            ty: &self.module.types[self.member.ty],
        }
    }

    pub fn layout(&self) -> TypeLayout<'a> {
        TypeLayout {
            module: self.module,
            ty: &self.module.types[self.member.ty],
        }
    }

    pub fn offset(&self) -> u32 {
        self.member.offset
    }
}


pub struct EntryPointReflection<'a> {
    module_info: &'a ModuleInfo,
    entry_point: &'a naga::EntryPoint,
    index: usize,
}

impl<'a> EntryPointReflection<'a> {
    pub fn name(&self) -> &str {
        &self.entry_point.name
    }

    pub fn stage(&self) -> naga::ShaderStage {
        self.entry_point.stage
    }

    pub fn workgroup_size(&self) -> [u32; 3] {
        self.entry_point.workgroup_size
    }

    pub fn uses_parameter(&self, param: &ParameterReflection) -> bool {
        let ep_info = self.module_info.get_entry_point(self.index);
        let usage = ep_info[param.var_handle()];
        !usage.is_empty()
    }

}


#[derive(Debug, Clone)]
pub struct ShaderDiff {
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub retained: Vec<String>,
    pub type_changed: Vec<(String, TypeKind, TypeKind)>,
}


fn vector_size_to_u32(size: &VectorSize) -> u32 {
    match size {
        VectorSize::Bi => 2,
        VectorSize::Tri => 3,
        VectorSize::Quad => 4,
    }
}

fn vector_size_to_u64(size: &VectorSize) -> u64 {
    vector_size_to_u32(size) as u64
}

pub(crate) fn type_size(module: &naga::Module, ty: &naga::Type) -> u64 {
    match &ty.inner {
        TypeInner::Scalar(scalar) => scalar.width as u64,
        TypeInner::Vector { size, scalar } => scalar.width as u64 * vector_size_to_u64(size),
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => scalar.width as u64 * vector_size_to_u64(columns) * vector_size_to_u64(rows),
        TypeInner::Atomic(scalar) => scalar.width as u64,
        TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => 8,
        TypeInner::Array {
            base, size, stride, ..
        } => {
            let base_size = type_size(module, &module.types[*base]);
            let count = match size {
                ArraySize::Constant(count) => count.get() as u64,
                ArraySize::Pending(_) | ArraySize::Dynamic => 0,
            };
            let element_size = std::cmp::max(base_size, *stride as u64);
            count * element_size
        }
        TypeInner::Struct { members, .. } => {
            let mut offset = 0;
            for member in members {
                let member_size = type_size(module, &module.types[member.ty]);
                offset = align_to(offset, 16);
                offset += member_size;
            }
            align_to(offset, 16)
        }
        TypeInner::Image { .. }
        | TypeInner::Sampler { .. }
        | TypeInner::BindingArray { .. }
        | TypeInner::RayQuery { .. } => 0,
        TypeInner::AccelerationStructure { .. } => 8,
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

fn texture_sample_type(image_class: &ImageClass) -> (bool, TextureSampleType) {
    match image_class {
        ImageClass::Sampled { multi, kind } => match kind {
            ScalarKind::Sint => (*multi, TextureSampleType::Sint),
            ScalarKind::Uint => (*multi, TextureSampleType::Uint),
            ScalarKind::Float => (*multi, TextureSampleType::Float { filterable: true }),
            _ => panic!("Unsupported scalar kind {:?}", kind),
        },
        ImageClass::Depth { multi } => (*multi, TextureSampleType::Depth),
        ImageClass::Storage { .. } => (false, TextureSampleType::Float { filterable: false }),
        ImageClass::External => (false, TextureSampleType::Float { filterable: true }),
    }
}

fn entry_builder(
    variable: &GlobalVariable,
    module: &naga::Module,
    ty: &naga::Type,
) -> BindGroupLayoutEntryBuilder {
    match &ty.inner {
        TypeInner::Image {
            dim,
            arrayed,
            class,
        } => match class {
            ImageClass::Storage { format, access } => {
                let access = if access.contains(StorageAccess::LOAD | StorageAccess::STORE) {
                    StorageTextureAccess::ReadWrite
                } else if access.contains(StorageAccess::STORE) {
                    StorageTextureAccess::WriteOnly
                } else if access.contains(StorageAccess::LOAD) {
                    StorageTextureAccess::ReadOnly
                } else {
                    panic!("Unsupported storage access {:?}", access)
                };
                let format = convert_storage_format(*format);
                let view_dimension = convert_image_dimension(*dim, *arrayed);
                match view_dimension {
                    TextureViewDimension::D2 => texture_storage_2d(format, access),
                    _ => panic!("Unsupported storage texture dimension {:?}", view_dimension),
                }
            }
            _ => {
                let (multi, sample_type) = texture_sample_type(class);
                match dim {
                    ImageDimension::D1 => match (arrayed, multi) {
                        (false, false) => texture_1d(sample_type),
                        (false, true) => panic!("Unsupported 1D multisampled image"),
                        (true, false) => panic!("Unsupported 1D array image"),
                        (true, true) => panic!("Unsupported 1D array multisampled image"),
                    },
                    ImageDimension::D2 => match (arrayed, multi) {
                        (false, false) => texture_2d(sample_type),
                        (false, true) => texture_2d_multisampled(sample_type),
                        (true, false) => texture_2d_array(sample_type),
                        (true, true) => texture_2d_array_multisampled(sample_type),
                    },
                    ImageDimension::D3 => match (arrayed, multi) {
                        (false, false) => texture_3d(sample_type),
                        (false, true) => texture_3d_multisampled(sample_type),
                        (true, false) => panic!("Unsupported 3D array image"),
                        (true, true) => panic!("Unsupported 3D array multisampled image"),
                    },
                    ImageDimension::Cube => match (arrayed, multi) {
                        (false, false) => texture_cube(sample_type),
                        (false, true) => texture_cube_multisampled(sample_type),
                        (true, false) => texture_cube_array(sample_type),
                        (true, true) => texture_cube_array_multisampled(sample_type),
                    },
                }
            }
        },
        TypeInner::Sampler { comparison } => {
            let binding_type = if *comparison {
                SamplerBindingType::Comparison
            } else {
                SamplerBindingType::Filtering
            };
            sampler(binding_type)
        }
        _ => {
            let size = type_size(module, ty);
            match variable.space {
                AddressSpace::Uniform => uniform_buffer_sized(false, NonZeroU64::new(size)),
                AddressSpace::Storage { access } => {
                    if access.contains(StorageAccess::LOAD | StorageAccess::STORE) {
                        storage_buffer_sized(false, NonZeroU64::new(size))
                    } else if access.contains(StorageAccess::LOAD) {
                        storage_buffer_read_only_sized(false, NonZeroU64::new(size))
                    } else {
                        panic!("Unsupported storage access {:?}", access)
                    }
                }
                _ => unimplemented!(),
            }
        }
    }
}

fn convert_storage_format(format: StorageFormat) -> TextureFormat {
    match format {
        StorageFormat::Rgba8Unorm => TextureFormat::Rgba8Unorm,
        _ => panic!("Unsupported storage format {:?}", format),
    }
}

fn convert_image_dimension(dim: ImageDimension, arrayed: bool) -> TextureViewDimension {
    match (dim, arrayed) {
        (ImageDimension::D1, false) => TextureViewDimension::D1,
        (ImageDimension::D2, false) => TextureViewDimension::D2,
        (ImageDimension::D2, true) => TextureViewDimension::D2Array,
        (ImageDimension::D3, false) => TextureViewDimension::D3,
        (ImageDimension::Cube, false) => TextureViewDimension::Cube,
        (ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
        _ => panic!(
            "Unsupported image dimension {:?} with arrayed {:?}",
            dim, arrayed
        ),
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use naga::{Handle, Module, ScalarKind, Type, TypeInner, VectorSize};
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
        assert_eq!(type_size(&module, &module.types[i32_type]), 4);

        let f64_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Float,
                width: 8,
            }),
        );
        assert_eq!(type_size(&module, &module.types[f64_type]), 8);

        let bool_type = add_type(
            &mut module,
            TypeInner::Scalar(naga::Scalar {
                kind: ScalarKind::Bool,
                width: 1,
            }),
        );
        assert_eq!(type_size(&module, &module.types[bool_type]), 1);
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
        assert_eq!(type_size(&module, &module.types[vec2_type]), 8);

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
        assert_eq!(type_size(&module, &module.types[vec4_type]), 16);
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
        assert_eq!(type_size(&module, &module.types[mat3x3_type]), 36);
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
        assert_eq!(type_size(&module, &module.types[atomic_i32_type]), 4);
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
        assert_eq!(type_size(&module, &module.types[fixed_array_type]), 40);

        let dynamic_array_type = add_type(
            &mut module,
            TypeInner::Array {
                base: f32_type,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        );
        assert_eq!(type_size(&module, &module.types[dynamic_array_type]), 0);
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
        assert_eq!(type_size(&module, &module.types[struct_type]), 32);
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
        assert_eq!(type_size(&module, &module.types[image_type]), 0);

        let sampler_type = add_type(&mut module, TypeInner::Sampler { comparison: false });
        assert_eq!(type_size(&module, &module.types[sampler_type]), 0);

        let accel_struct_type = add_type(
            &mut module,
            TypeInner::AccelerationStructure {
                vertex_return: false,
            },
        );
        assert_eq!(type_size(&module, &module.types[accel_struct_type]), 8);
    }


    use bevy::render::render_resource::{BindingType, BufferBindingType};

    fn reflect(wgsl: &str) -> ShaderReflection {
        let module = naga::front::wgsl::parse_str(wgsl).unwrap();
        ShaderReflection::new(module).unwrap()
    }

    #[test]
    fn layout_vec4() {
        let reflection = reflect(
            r#"
            @group(2) @binding(1) var<uniform> color: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return color; }
        "#,
        );

        let layouts = reflection.bind_group_layout(2);
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].binding, 1);
        assert_eq!(layouts[0].visibility, ShaderStages::FRAGMENT);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
            }
        );
    }

    #[test]
    fn layout_texture() {
        let reflection = reflect(
            r#"
            @group(2) @binding(1) var my_texture: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(my_texture, vec2<i32>(0, 0), 0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(2);
        assert_eq!(layouts.len(), 1);
        assert_eq!(layouts[0].binding, 1);
        assert_eq!(layouts[0].visibility, ShaderStages::FRAGMENT);
        assert_eq!(
            layouts[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                multisampled: false,
                view_dimension: TextureViewDimension::D2,
            }
        );
    }

    #[test]
    fn layout_scalar_int() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: i32;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(f32(value), 0.0, 0.0, 1.0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            }
        );
        assert_eq!(layouts[0].visibility, ShaderStages::FRAGMENT);
    }

    #[test]
    fn layout_vector_float() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> vector: vec3<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(vector, 1.0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(12).unwrap()),
            }
        );
    }

    #[test]
    fn layout_matrix() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> matrix: mat4x4<f32>;
            @vertex fn main(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return matrix * pos;
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(64).unwrap()),
            }
        );
    }

    #[test]
    fn layout_array() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> my_array: array<vec4<f32>, 10>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return my_array[0];
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(160).unwrap()),
            }
        );
    }

    #[test]
    fn layout_struct() {
        let reflection = reflect(
            r#"
            struct MyStruct {
                a: f32,
                b: vec3<f32>,
            }
            @group(0) @binding(0) var<uniform> my_struct: MyStruct;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(my_struct.b, my_struct.a);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(32).unwrap()),
            }
        );
    }

    #[test]
    fn layout_sampler() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_sampler: sampler;
            @group(0) @binding(1) var my_tex: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(my_tex, my_sampler, vec2<f32>(0.0, 0.0));
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        let sampler_layout = layouts.iter().find(|l| l.binding == 0).unwrap();
        assert_eq!(
            sampler_layout.ty,
            BindingType::Sampler(SamplerBindingType::Filtering)
        );
    }

    #[test]
    fn layout_comparison_sampler() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_sampler: sampler_comparison;
            @group(0) @binding(1) var my_tex: texture_depth_2d;
            @fragment fn main() -> @location(0) vec4<f32> {
                let d = textureSampleCompare(my_tex, my_sampler, vec2<f32>(0.0, 0.0), 0.5);
                return vec4<f32>(d, 0.0, 0.0, 1.0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        let sampler_layout = layouts.iter().find(|l| l.binding == 0).unwrap();
        assert_eq!(
            sampler_layout.ty,
            BindingType::Sampler(SamplerBindingType::Comparison)
        );
    }

    #[test]
    fn layout_texture_1d() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_texture: texture_1d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(my_texture, 0, 0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D1,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_texture_3d() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_texture: texture_3d<u32>;
            @fragment fn main() -> @location(0) vec4<u32> {
                return textureLoad(my_texture, vec3<i32>(0, 0, 0), 0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_texture_cube_array() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_texture: texture_cube_array<f32>;
            @group(0) @binding(1) var my_sampler: sampler;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(my_texture, my_sampler, vec3<f32>(0.0, 0.0, 1.0), 0);
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        let tex_layout = layouts.iter().find(|l| l.binding == 0).unwrap();
        assert_eq!(
            tex_layout.ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::CubeArray,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_storage_texture() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_texture: texture_storage_2d<rgba8unorm, write>;
            @compute @workgroup_size(1) fn main() {
                textureStore(my_texture, vec2<i32>(0, 0), vec4<f32>(1.0));
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            }
        );
    }

    #[test]
    fn layout_storage_buffer() {
        let reflection = reflect(
            r#"
            struct Data {
                values: array<f32>,
            }
            @group(0) @binding(0) var<storage, read_write> my_buffer: Data;
            @compute @workgroup_size(1) fn main() {
                my_buffer.values[0] = 1.0;
            }
        "#,
        );

        let layouts = reflection.bind_group_layout(0);
        assert_eq!(
            layouts[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        );
    }


    #[test]
    fn visibility_fragment_only() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return color; }
        "#,
        );

        let param = reflection.parameter("color").unwrap();
        assert_eq!(param.visibility(), ShaderStages::FRAGMENT);
    }

    #[test]
    fn visibility_vertex_only() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
            @vertex fn main(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return mvp * pos;
            }
        "#,
        );

        let param = reflection.parameter("mvp").unwrap();
        assert_eq!(param.visibility(), ShaderStages::VERTEX);
    }

    #[test]
    fn visibility_vertex_and_fragment() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;
            @group(0) @binding(1) var<uniform> color: vec4<f32>;

            @vertex fn vs(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return mvp * pos;
            }

            @fragment fn fs() -> @location(0) vec4<f32> {
                return color;
            }
        "#,
        );

        let mvp = reflection.parameter("mvp").unwrap();
        assert_eq!(mvp.visibility(), ShaderStages::VERTEX);

        let color = reflection.parameter("color").unwrap();
        assert_eq!(color.visibility(), ShaderStages::FRAGMENT);
    }

    #[test]
    fn visibility_shared_parameter() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> tint: vec4<f32>;

            struct VertexOut {
                @builtin(position) pos: vec4<f32>,
                @location(0) color: vec4<f32>,
            }

            @vertex fn vs(@location(0) pos: vec4<f32>) -> VertexOut {
                var out: VertexOut;
                out.pos = pos;
                out.color = tint;
                return out;
            }

            @fragment fn fs(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
                return color * tint;
            }
        "#,
        );

        let tint = reflection.parameter("tint").unwrap();
        assert_eq!(tint.visibility(), ShaderStages::VERTEX_FRAGMENT);
    }

    #[test]
    fn visibility_no_entry_points_skips() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: f32;
        "#,
        );

        assert_eq!(reflection.parameter_count(), 0);
        assert!(reflection.parameter("value").is_none());
    }


    #[test]
    fn diff_added_parameter() {
        let old = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return color; }
        "#,
        );
        let new = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @group(0) @binding(1) var<uniform> brightness: f32;
            @fragment fn main() -> @location(0) vec4<f32> { return color * brightness; }
        "#,
        );

        let diff = old.diff(&new);
        assert_eq!(diff.added, vec!["brightness"]);
        assert!(diff.removed.is_empty());
        assert_eq!(diff.retained, vec!["color"]);
        assert!(diff.type_changed.is_empty());
    }

    #[test]
    fn diff_removed_parameter() {
        let old = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @group(0) @binding(1) var<uniform> brightness: f32;
            @fragment fn main() -> @location(0) vec4<f32> { return color * brightness; }
        "#,
        );
        let new = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return color; }
        "#,
        );

        let diff = old.diff(&new);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed, vec!["brightness"]);
        assert_eq!(diff.retained, vec!["color"]);
    }

    #[test]
    fn diff_type_changed() {
        let old = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: f32;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(value, 0.0, 0.0, 1.0);
            }
        "#,
        );
        let new = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return value; }
        "#,
        );

        let diff = old.diff(&new);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.retained, vec!["value"]);
        assert_eq!(diff.type_changed.len(), 1);
        assert_eq!(diff.type_changed[0].0, "value");
        assert_eq!(diff.type_changed[0].1, TypeKind::Scalar);
        assert_eq!(diff.type_changed[0].2, TypeKind::Vector);
    }


    #[test]
    fn type_reflection_scalar() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: f32;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(value, 0.0, 0.0, 1.0);
            }
        "#,
        );

        let param = reflection.parameter("value").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Scalar);
        assert_eq!(ty.scalar_type(), Some(ScalarType::Float));
        assert_eq!(ty.scalar_width(), Some(4));
        assert!(ty.vector_size().is_none());
    }

    #[test]
    fn type_reflection_vector() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> v: vec3<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(v, 1.0);
            }
        "#,
        );

        let param = reflection.parameter("v").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Vector);
        assert_eq!(ty.scalar_type(), Some(ScalarType::Float));
        assert_eq!(ty.vector_size(), Some(3));
    }

    #[test]
    fn type_reflection_matrix() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> m: mat4x4<f32>;
            @vertex fn main(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return m * pos;
            }
        "#,
        );

        let param = reflection.parameter("m").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Matrix);
        assert_eq!(ty.columns(), Some(4));
        assert_eq!(ty.rows(), Some(4));
        assert_eq!(ty.scalar_type(), Some(ScalarType::Float));
    }

    #[test]
    fn type_reflection_struct_fields() {
        let reflection = reflect(
            r#"
            struct MyStruct {
                x: f32,
                y: vec3<f32>,
            }
            @group(0) @binding(0) var<uniform> s: MyStruct;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(s.y, s.x);
            }
        "#,
        );

        let param = reflection.parameter("s").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Struct);
        assert_eq!(ty.name(), Some("MyStruct"));

        let fields = ty.fields().unwrap();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name(), Some("x"));
        assert_eq!(fields[0].ty().kind(), TypeKind::Scalar);
        assert_eq!(fields[1].name(), Some("y"));
        assert_eq!(fields[1].ty().kind(), TypeKind::Vector);

        let y_field = ty.field("y").unwrap();
        assert_eq!(y_field.ty().vector_size(), Some(3));
    }

    #[test]
    fn type_reflection_array() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> arr: array<vec4<f32>, 5>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return arr[0];
            }
        "#,
        );

        let param = reflection.parameter("arr").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Array);
        assert_eq!(ty.element_count(), Some(5));

        let elem_ty = ty.element_type().unwrap();
        assert_eq!(elem_ty.kind(), TypeKind::Vector);
        assert_eq!(elem_ty.scalar_type(), Some(ScalarType::Float));
    }

    #[test]
    fn type_reflection_image() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_tex: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureLoad(my_tex, vec2<i32>(0, 0), 0);
            }
        "#,
        );

        let param = reflection.parameter("my_tex").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Image);
        assert_eq!(ty.image_dimension(), Some(ImageDimension::D2));
        assert_eq!(param.category(), ParameterCategory::Texture);
    }

    #[test]
    fn type_reflection_sampler() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_sampler: sampler;
            @group(0) @binding(1) var my_tex: texture_2d<f32>;
            @fragment fn main() -> @location(0) vec4<f32> {
                return textureSample(my_tex, my_sampler, vec2<f32>(0.0, 0.0));
            }
        "#,
        );

        let param = reflection.parameter("my_sampler").unwrap();
        let ty = param.ty();
        assert_eq!(ty.kind(), TypeKind::Sampler);
        assert!(!ty.is_comparison_sampler());
        assert_eq!(param.category(), ParameterCategory::Sampler);
    }

    #[test]
    fn type_reflection_comparison_sampler() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var my_sampler: sampler_comparison;
            @group(0) @binding(1) var my_tex: texture_depth_2d;
            @fragment fn main() -> @location(0) vec4<f32> {
                let d = textureSampleCompare(my_tex, my_sampler, vec2<f32>(0.0, 0.0), 0.5);
                return vec4<f32>(d, 0.0, 0.0, 1.0);
            }
        "#,
        );

        let param = reflection.parameter("my_sampler").unwrap();
        assert!(param.ty().is_comparison_sampler());
    }


    #[test]
    fn parameter_category_uniform() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> value: f32;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(value, 0.0, 0.0, 1.0);
            }
        "#,
        );

        let param = reflection.parameter("value").unwrap();
        assert_eq!(param.category(), ParameterCategory::Uniform);
    }

    #[test]
    fn parameter_category_storage() {
        let reflection = reflect(
            r#"
            struct Data { values: array<f32> }
            @group(0) @binding(0) var<storage, read_write> data: Data;
            @compute @workgroup_size(1) fn main() {
                data.values[0] = 1.0;
            }
        "#,
        );

        let param = reflection.parameter("data").unwrap();
        assert_eq!(
            param.category(),
            ParameterCategory::Storage { read_only: false }
        );
    }

    #[test]
    fn parameter_category_storage_read_only() {
        let reflection = reflect(
            r#"
            struct Data { values: array<f32> }
            @group(0) @binding(0) var<storage, read> data: Data;
            @compute @workgroup_size(1) fn main() {
                let x = data.values[0];
                _ = x;
            }
        "#,
        );

        let param = reflection.parameter("data").unwrap();
        assert_eq!(
            param.category(),
            ParameterCategory::Storage { read_only: true }
        );
    }


    #[test]
    fn entry_point_reflection() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> color: vec4<f32>;

            @vertex fn vs(@location(0) pos: vec4<f32>) -> @builtin(position) vec4<f32> {
                return pos;
            }

            @fragment fn fs() -> @location(0) vec4<f32> {
                return color;
            }
        "#,
        );

        let eps: Vec<_> = reflection.entry_points().collect();
        assert_eq!(eps.len(), 2);

        assert_eq!(eps[0].name(), "vs");
        assert_eq!(eps[0].stage(), naga::ShaderStage::Vertex);

        assert_eq!(eps[1].name(), "fs");
        assert_eq!(eps[1].stage(), naga::ShaderStage::Fragment);

        let color = reflection.parameter("color").unwrap();
        assert!(!eps[0].uses_parameter(&color));
        assert!(eps[1].uses_parameter(&color));
    }


    #[test]
    fn type_layout_size() {
        let reflection = reflect(
            r#"
            @group(0) @binding(0) var<uniform> v: vec4<f32>;
            @fragment fn main() -> @location(0) vec4<f32> { return v; }
        "#,
        );

        let param = reflection.parameter("v").unwrap();
        assert_eq!(param.layout().size(), 16);
    }

    #[test]
    fn type_layout_struct_size() {
        let reflection = reflect(
            r#"
            struct MyStruct {
                a: f32,
                b: vec3<f32>,
            }
            @group(0) @binding(0) var<uniform> s: MyStruct;
            @fragment fn main() -> @location(0) vec4<f32> {
                return vec4<f32>(s.b, s.a);
            }
        "#,
        );

        let param = reflection.parameter("s").unwrap();
        assert_eq!(param.layout().size(), 32);
    }
}
