use crate::reflect::{type_align, type_size};
use bevy::asset::Handle;
use bevy::log::warn;
use bevy::math::{IVec2, IVec3, IVec4, Mat4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};
use bevy::prelude::Image;
use bevy::reflect::{FromReflect, PartialReflect, ReflectRef};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    BufferInitDescriptor, BufferUsages, OwnedBindingResource, Sampler, SamplerBindingType,
    TextureView, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use naga::{ImageDimension, ScalarKind, VectorSize};

pub fn find_field<'a>(
    reflected: &'a dyn PartialReflect,
    field_name: &str,
) -> Option<&'a dyn PartialReflect> {
    let ReflectRef::Struct(reflect_struct) = reflected.reflect_ref() else {
        warn!("Cannot reflect struct for binding");
        return None;
    };
    reflect_struct.field(field_name)
}

pub fn generate_image_binding(ty: &naga::Type, image: &GpuImage) -> OwnedBindingResource {
    match &ty.inner {
        naga::TypeInner::Image { dim, arrayed, .. } => {
            let view_dimension = match (dim, arrayed) {
                (ImageDimension::D1, false) => TextureViewDimension::D1,
                (ImageDimension::D2, false) => TextureViewDimension::D2,
                (ImageDimension::D2, true) => TextureViewDimension::D2Array,
                (ImageDimension::D3, false) => TextureViewDimension::D3,
                (ImageDimension::Cube, false) => TextureViewDimension::Cube,
                (ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
                _ => TextureViewDimension::D2,
            };
            OwnedBindingResource::TextureView(view_dimension, image.texture_view.clone())
        }
        naga::TypeInner::Sampler { comparison } => {
            let binding_type = if *comparison {
                SamplerBindingType::Comparison
            } else {
                SamplerBindingType::Filtering
            };
            OwnedBindingResource::Sampler(binding_type, image.sampler.clone())
        }
        _ => panic!("generate_image_binding called with non-image/sampler type"),
    }
}

pub fn generate_texture_view_binding(ty: &naga::Type, view: &TextureView) -> OwnedBindingResource {
    let naga::TypeInner::Image { dim, arrayed, .. } = &ty.inner else {
        panic!("generate_texture_view_binding called with non-image type");
    };
    let view_dimension = match (dim, arrayed) {
        (ImageDimension::D1, false) => TextureViewDimension::D1,
        (ImageDimension::D2, false) => TextureViewDimension::D2,
        (ImageDimension::D2, true) => TextureViewDimension::D2Array,
        (ImageDimension::D3, false) => TextureViewDimension::D3,
        (ImageDimension::Cube, false) => TextureViewDimension::Cube,
        (ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
        _ => TextureViewDimension::D2,
    };
    OwnedBindingResource::TextureView(view_dimension, view.clone())
}

pub fn generate_sampler_binding(ty: &naga::Type, sampler: &Sampler) -> OwnedBindingResource {
    let comparison = matches!(
        ty.inner,
        naga::TypeInner::Sampler {
            comparison: true,
            ..
        }
    );
    let binding_type = if comparison {
        SamplerBindingType::Comparison
    } else {
        SamplerBindingType::Filtering
    };
    OwnedBindingResource::Sampler(binding_type, sampler.clone())
}

pub fn generate_binding_resource(
    field_value: &dyn PartialReflect,
    module: &naga::Module,
    ty: &naga::Type,
    render_device: &RenderDevice,
    gpu_images: &RenderAssets<GpuImage>,
    gpu_buffers: &RenderAssets<GpuShaderBuffer>,
) -> OwnedBindingResource {
    match &ty.inner {
        naga::TypeInner::Image { dim, arrayed, .. } => {
            let handle = field_value
                .try_downcast_ref::<Handle<Image>>()
                .expect("Field value is not an image");
            let image = gpu_images.get(handle).unwrap();
            let view_dimension = match (dim, arrayed) {
                (ImageDimension::D1, false) => TextureViewDimension::D1,
                (ImageDimension::D2, false) => TextureViewDimension::D2,
                (ImageDimension::D2, true) => TextureViewDimension::D2Array,
                (ImageDimension::D3, false) => TextureViewDimension::D3,
                (ImageDimension::Cube, false) => TextureViewDimension::Cube,
                (ImageDimension::Cube, true) => TextureViewDimension::CubeArray,
                _ => TextureViewDimension::D2,
            };
            OwnedBindingResource::TextureView(view_dimension, image.texture_view.clone())
        }
        naga::TypeInner::Sampler { comparison } => {
            let handle = field_value
                .try_downcast_ref::<Handle<Image>>()
                .expect("Field value is not an image");
            let image = gpu_images.get(handle).unwrap();
            let binding_type = if *comparison {
                SamplerBindingType::Comparison
            } else {
                SamplerBindingType::Filtering
            };
            OwnedBindingResource::Sampler(binding_type, image.sampler.clone())
        }
        _ => {
            if let Some(handle) = field_value.try_downcast_ref::<Handle<ShaderBuffer>>() {
                if let Some(gpu_buffer) = gpu_buffers.get(handle) {
                    return OwnedBindingResource::Buffer(gpu_buffer.buffer.clone());
                }
            }

            let mut buffer: Vec<u8> = Vec::new();
            write_to_buffer(field_value, module, ty, &mut buffer);
            OwnedBindingResource::Buffer(render_device.create_buffer_with_data(
                &BufferInitDescriptor {
                    label: None,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                    contents: &buffer,
                },
            ))
        }
    }
}

/// Encode a reflected value into raw bytes following WGSL uniform-space memory
/// layout. Compound types (struct, array) insert padding between elements per
/// the alignment rules in [`type_align`].
///
/// `buffer` is appended to; the caller decides how to use the resulting bytes.
pub fn write_to_buffer(
    field_value: &dyn PartialReflect,
    module: &naga::Module,
    ty: &naga::Type,
    buffer: &mut Vec<u8>,
) {
    match &ty.inner {
        naga::TypeInner::Scalar(scalar) => match scalar.kind {
            ScalarKind::Sint => {
                let v = field_value.try_downcast_ref::<i32>().unwrap();
                buffer.extend_from_slice(&v.to_le_bytes());
            }
            ScalarKind::Uint => {
                let v = field_value.try_downcast_ref::<u32>().unwrap();
                buffer.extend_from_slice(&v.to_le_bytes());
            }
            ScalarKind::Float => {
                let v = field_value.try_downcast_ref::<f32>().unwrap();
                buffer.extend_from_slice(&v.to_le_bytes());
            }
            ScalarKind::Bool => {
                let v = field_value
                    .try_downcast_ref::<u32>()
                    .copied()
                    .or_else(|| {
                        field_value
                            .try_downcast_ref::<bool>()
                            .map(|b| if *b { 1u32 } else { 0u32 })
                    })
                    .unwrap();
                buffer.extend_from_slice(&v.to_le_bytes());
            }
            _ => panic!("Unsupported scalar type: {:?}", ty),
        },
        naga::TypeInner::Vector { size, scalar } => match (size, scalar.kind) {
            (VectorSize::Bi, ScalarKind::Float) => {
                let v = Vec2::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
            }
            (VectorSize::Tri, ScalarKind::Float) => {
                let v = Vec3::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
                buffer.extend_from_slice(&v.z.to_le_bytes());
                // Note: vec3 has size 12 / align 16. The 4-byte tail padding is
                // inserted by the parent (struct member alignment, array stride)
                // — never inline here.
            }
            (VectorSize::Quad, ScalarKind::Float) => {
                let v = Vec4::from_reflect(field_value).unwrap();
                for c in [v.x, v.y, v.z, v.w] {
                    buffer.extend_from_slice(&c.to_le_bytes());
                }
            }
            (VectorSize::Bi, ScalarKind::Sint) => {
                let v = IVec2::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
            }
            (VectorSize::Tri, ScalarKind::Sint) => {
                let v = IVec3::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
                buffer.extend_from_slice(&v.z.to_le_bytes());
            }
            (VectorSize::Quad, ScalarKind::Sint) => {
                let v = IVec4::from_reflect(field_value).unwrap();
                for c in [v.x, v.y, v.z, v.w] {
                    buffer.extend_from_slice(&c.to_le_bytes());
                }
            }
            (VectorSize::Bi, ScalarKind::Uint) => {
                let v = UVec2::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
            }
            (VectorSize::Tri, ScalarKind::Uint) => {
                let v = UVec3::from_reflect(field_value).unwrap();
                buffer.extend_from_slice(&v.x.to_le_bytes());
                buffer.extend_from_slice(&v.y.to_le_bytes());
                buffer.extend_from_slice(&v.z.to_le_bytes());
            }
            (VectorSize::Quad, ScalarKind::Uint) => {
                let v = UVec4::from_reflect(field_value).unwrap();
                for c in [v.x, v.y, v.z, v.w] {
                    buffer.extend_from_slice(&c.to_le_bytes());
                }
            }
            _ => panic!("Unsupported vector type: {:?}", ty),
        },
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => match (columns, rows, scalar.kind) {
            (VectorSize::Quad, VectorSize::Quad, ScalarKind::Float) => {
                let m = field_value.try_downcast_ref::<Mat4>().unwrap();
                let cols = [m.x_axis, m.y_axis, m.z_axis, m.w_axis];
                for col in cols {
                    for c in [col.x, col.y, col.z, col.w] {
                        buffer.extend_from_slice(&c.to_le_bytes());
                    }
                }
            }
            _ => panic!("Unsupported matrix type: {:?}", ty),
        },
        naga::TypeInner::Array { base, .. } => {
            let ReflectRef::Array(array) = field_value.reflect_ref() else {
                panic!("Field value is not an array");
            };
            let base_ty = &module.types[*base];
            let element_size = type_size(module, base_ty);
            let element_align = type_align(module, base_ty);
            let stride = align_up(element_size, element_align);
            let array_start = buffer.len();
            for (i, item) in array.iter().enumerate() {
                let target = array_start + (i * stride as usize);
                pad_to(buffer, target);
                write_to_buffer(item, module, base_ty, buffer);
            }
        }
        naga::TypeInner::Struct { members, .. } => {
            let ReflectRef::Struct(reflect_struct) = field_value.reflect_ref() else {
                panic!("Field value is not a struct");
            };
            let struct_start = buffer.len();
            let struct_size = type_size(module, ty);
            for member in members {
                let Some(name) = member.name.as_ref() else {
                    panic!("Struct member has no name");
                };
                let member_ty = &module.types[member.ty];
                let member_align = type_align(module, member_ty);
                let cur = (buffer.len() - struct_start) as u64;
                let target = align_up(cur, member_align);
                pad_to(buffer, struct_start + target as usize);
                if let Some(field) = reflect_struct.field(name) {
                    write_to_buffer(field, module, member_ty, buffer);
                } else {
                    panic!("Struct field not found: {:?}", member.name);
                }
            }
            // Pad the struct out to its computed total size so the resulting
            // binding matches `min_binding_size`.
            pad_to(buffer, struct_start + struct_size as usize);
        }
        _ => {}
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

fn pad_to(buffer: &mut Vec<u8>, target_len: usize) {
    while buffer.len() < target_len {
        buffer.push(0);
    }
}
