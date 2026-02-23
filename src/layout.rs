use crate::module::{type_size, Binding};
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
use naga::{
    AddressSpace, ImageClass, ImageDimension, ScalarKind, StorageAccess, StorageFormat, TypeInner,
};
use std::num::NonZeroU64;

pub fn layout(module: &naga::Module, visibility: ShaderStages) -> Vec<BindGroupLayoutEntry> {
    let bindings = crate::module::bindings(module);
    bindings
        .map(|binding| entry_builder(module, &binding).build(binding.binding, visibility))
        .collect()
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
        ImageClass::Storage { .. } => {
            (false, TextureSampleType::Float { filterable: false })
        }
        ImageClass::External => {
            (false, TextureSampleType::Float { filterable: true })
        }
    }
}

fn entry_builder(module: &naga::Module, binding: &Binding) -> BindGroupLayoutEntryBuilder {
    match &binding.ty.inner {
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
                    TextureViewDimension::D2 => return texture_storage_2d(format, access),
                    _ => panic!("Unsupported storage texture dimension {:?}", view_dimension),
                }
            }
            _ => {
                let (multi, sample_type) = texture_sample_type(class);
                return match dim {
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
                };
            }
        },
        TypeInner::Sampler { comparison } => {
            let binding_type = if *comparison {
                SamplerBindingType::Comparison
            } else {
                // TODO: handle filtering
                SamplerBindingType::Filtering
            };

            return sampler(binding_type);
        }
        TypeInner::Scalar(_scalar) => {}
        _ => {}
    };

    let size = type_size(module, &binding.ty);

    match binding.variable.space {
        AddressSpace::Uniform => uniform_buffer_sized(false, NonZeroU64::new(size)),
        AddressSpace::Storage { access } => match access {
            x if x == StorageAccess::LOAD | StorageAccess::STORE => {
                storage_buffer_sized(false, NonZeroU64::new(size))
            }
            x if x == StorageAccess::LOAD => {
                storage_buffer_read_only_sized(false, NonZeroU64::new(size))
            }
            _ => panic!("Unsupported storage access {:?}", access),
        },
        _ => unimplemented!(),
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
    use bevy::render::render_resource::{
        BindingType, BufferBindingType, StorageTextureAccess, TextureFormat, TextureViewDimension,
    };

    #[test]
    fn layout_vec4() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(2) @binding(1) var<uniform> color: vec4<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].binding, 1);
        assert_eq!(bindings[0].visibility, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(16).unwrap()),
            }
        );
    }

    #[test]
    fn layout_texture() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(2) @binding(1) var my_texture: texture_2d<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].binding, 1);
        assert_eq!(bindings[0].visibility, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                multisampled: false,
                view_dimension: bevy::render::render_resource::TextureViewDimension::D2,
            }
        );
    }

    #[test]
    fn layout_scalar_int() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var<uniform> value: i32;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::VERTEX);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            }
        );
    }

    #[test]
    fn layout_vector_float() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var<uniform> vector: vec3<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::VERTEX);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(12).unwrap()),
            }
        );
    }

    #[test]
    fn layout_matrix() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var<uniform> matrix: mat4x4<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::VERTEX);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(64).unwrap()),
            }
        );
    }

    #[test]
    fn layout_array() {
        let module = naga::front::wgsl::parse_str(
            r#"
        @group(0) @binding(0) var<uniform> my_array: array<f32, 10>;
    "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::VERTEX);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(40).unwrap()),
            }
        );
    }

    #[test]
    fn layout_struct() {
        let module = naga::front::wgsl::parse_str(
            r#"
            struct MyStruct {
                a: f32,
                b: vec3<f32>,
            }
            @group(0) @binding(0) var<uniform> my_struct: MyStruct;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::VERTEX);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(32).unwrap()),
            }
        );
    }

    #[test]
    fn layout_sampler() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_sampler: sampler;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Sampler(SamplerBindingType::Filtering)
        );
    }

    #[test]
    fn layout_comparison_sampler() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_sampler: sampler_comparison;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Sampler(SamplerBindingType::Comparison)
        );
    }

    #[test]
    fn layout_texture_1d() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_texture: texture_1d<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::D1,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_texture_3d() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_texture: texture_3d<u32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Uint,
                view_dimension: TextureViewDimension::D3,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_texture_cube_array() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_texture: texture_cube_array<f32>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::FRAGMENT);
        assert_eq!(
            bindings[0].ty,
            BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension: TextureViewDimension::CubeArray,
                multisampled: false,
            }
        );
    }

    #[test]
    fn layout_storage_texture() {
        let module = naga::front::wgsl::parse_str(
            r#"
            @group(0) @binding(0) var my_texture: texture_storage_2d<rgba8unorm, write>;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::COMPUTE);
        assert_eq!(
            bindings[0].ty,
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba8Unorm,
                view_dimension: TextureViewDimension::D2,
            }
        );
    }

    #[test]
    fn layout_storage_buffer() {
        let module = naga::front::wgsl::parse_str(
            r#"
            struct Data {
                values: array<f32>,
            }
            @group(0) @binding(0) var<storage, read_write> my_buffer: Data;
        "#,
        )
        .unwrap();

        let bindings = layout(&module, ShaderStages::COMPUTE);
        assert_eq!(
            bindings[0].ty,
            BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        );
    }
}
