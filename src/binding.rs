use bevy::asset::Handle;
use bevy::log::warn;
use bevy::math::{Mat4, Vec2, Vec3, Vec4};
use bevy::prelude::Image;
use bevy::reflect::{FromReflect, PartialReflect, ReflectRef};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::encase::UniformBuffer as EncaseUniformBuffer;
use bevy::render::render_resource::{
    BufferInitDescriptor, BufferUsages, OwnedBindingResource, SamplerBindingType,
    TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::GpuImage;
use naga::{ImageDimension, ScalarKind, VectorSize};

pub(crate) fn find_field<'a>(
    reflected: &'a dyn PartialReflect,
    field_name: &str,
) -> Option<&'a dyn PartialReflect> {
    let ReflectRef::Struct(reflect_struct) = reflected.reflect_ref() else {
        warn!("Cannot reflect struct for binding");
        return None;
    };
    reflect_struct.field(field_name)
}

pub(crate) fn generate_binding_resource(
    field_value: &dyn PartialReflect,
    module: &naga::Module,
    ty: &naga::Type,
    render_device: &RenderDevice,
    gpu_images: &RenderAssets<GpuImage>,
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
            let mut buffer = EncaseUniformBuffer::new(Vec::new());
            write_to_buffer(field_value, module, ty, &mut buffer);
            OwnedBindingResource::Buffer(render_device.create_buffer_with_data(
                &BufferInitDescriptor {
                    label: None,
                    usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
                    contents: buffer.as_ref(),
                },
            ))
        }
    }
}

pub(crate) fn write_to_buffer(
    field_value: &dyn PartialReflect,
    module: &naga::Module,
    ty: &naga::Type,
    buffer: &mut EncaseUniformBuffer<Vec<u8>>,
) {
    match &ty.inner {
        naga::TypeInner::Scalar(scalar) => match scalar.kind {
            ScalarKind::Sint => buffer
                .write(field_value.try_downcast_ref::<i32>().unwrap())
                .unwrap(),
            ScalarKind::Uint => buffer
                .write(field_value.try_downcast_ref::<u32>().unwrap())
                .unwrap(),
            ScalarKind::Float => buffer
                .write(field_value.try_downcast_ref::<f32>().unwrap())
                .unwrap(),
            ScalarKind::Bool => buffer
                .write(field_value.try_downcast_ref::<u32>().unwrap())
                .unwrap(),
            _ => panic!("Unsupported scalar type: {:?}", ty),
        },
        naga::TypeInner::Vector { size, scalar } => match (size, scalar.kind) {
            (VectorSize::Bi, ScalarKind::Float) => buffer
                .write(&Vec2::from_reflect(field_value).unwrap())
                .unwrap(),
            (VectorSize::Tri, ScalarKind::Float) => buffer
                .write(&Vec3::from_reflect(field_value).unwrap())
                .unwrap(),
            (VectorSize::Quad, ScalarKind::Float) => buffer
                .write(&Vec4::from_reflect(field_value).unwrap())
                .unwrap(),
            _ => panic!("Unsupported vector type: {:?}", ty),
        },
        naga::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => match (columns, rows, scalar.kind) {
            (VectorSize::Quad, VectorSize::Quad, ScalarKind::Float) => buffer
                .write(field_value.try_downcast_ref::<Mat4>().unwrap())
                .unwrap(),
            _ => panic!("Unsupported matrix type: {:?}", ty),
        },
        naga::TypeInner::Array { base, .. } => {
            let ReflectRef::Array(array) = field_value.reflect_ref() else {
                panic!("Field value is not an array");
            };
            let base = &module.types[*base];
            for item in array.iter() {
                write_to_buffer(item, module, base, buffer);
            }
        }
        naga::TypeInner::Struct { members, .. } => {
            let ReflectRef::Struct(reflect_struct) = field_value.reflect_ref() else {
                panic!("Field value is not a struct");
            };
            for member in members {
                let Some(name) = member.name.as_ref() else {
                    panic!("Struct member has no name");
                };

                if let Some(field) = reflect_struct.field(name) {
                    let member_ty = &module.types[member.ty];
                    write_to_buffer(field, module, member_ty, buffer);
                } else {
                    panic!("Struct field not found: {:?}", member.name);
                }
            }
        }
        _ => {}
    }
}
