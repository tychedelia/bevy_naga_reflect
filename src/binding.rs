use crate::module;
use bevy::asset::Handle;
use bevy::log::warn;
use bevy::math::{Mat4, Vec2, Vec3, Vec4};
use bevy::prelude::info;
use bevy::reflect::{reflect_trait, FromReflect, GetField, Reflect, ReflectRef, TypeRegistry};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::encase::private::WriteInto;
use bevy::render::render_resource::encase::UniformBuffer;
use bevy::render::render_resource::{
    BindingResource, Buffer, BufferInitDescriptor, BufferUsages, OwnedBindingResource, ShaderSize,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{GpuImage, Image};
use naga::{ScalarKind, VectorSize};
use std::sync::Arc;

pub fn bindings(
    module: &naga::Module,
    reflected: &dyn Reflect,
    render_device: &RenderDevice,
    gpu_images: &RenderAssets<GpuImage>,
) -> Vec<(u32, OwnedBindingResource)> {
    let mut bindings = Vec::new();

    let module_bindings = module::bindings(module);

    for binding in module_bindings {
        let ty = binding.ty;
        let Some(name) = binding.variable.name.as_ref() else {
            continue;
        };

        if let Some(field_value) = find_field(reflected, name) {
            let binding = binding.binding;
            let resource =
                generate_binding_resource(field_value, module, &ty, render_device, gpu_images);
            bindings.push((binding, resource));
        } else {
            warn!("Field not found in reflected type: {:?}", name);
        }
    }

    bindings
}

fn find_field<'a>(reflected: &'a dyn Reflect, field_name: &str) -> Option<&'a dyn Reflect> {
    let ReflectRef::Struct(reflect_struct) = reflected.reflect_ref() else {
        warn!("Cannot reflect struct for binding",);
        return None;
    };

    reflect_struct.field(field_name)
}

fn generate_binding_resource(
    field_value: &dyn Reflect,
    module: &naga::Module,
    ty: &naga::Type,
    render_device: &RenderDevice,
    gpu_images: &RenderAssets<GpuImage>,
) -> OwnedBindingResource {
    match &ty.inner {
        naga::TypeInner::Image { .. } => {
            let handle = field_value
                .try_downcast_ref::<Handle<Image>>()
                .expect("Field value is not an image");
            let image = gpu_images.get(handle).unwrap();
            OwnedBindingResource::TextureView(image.texture_view.clone())
        }
        naga::TypeInner::Sampler { .. } => {
            let handle = field_value
                .try_downcast_ref::<Handle<Image>>()
                .expect("Field value is not an image");
            let image = gpu_images.get(handle).unwrap();
            OwnedBindingResource::Sampler(image.sampler.clone())
        }
        _ => {
            let mut buffer = UniformBuffer::new(Vec::new());
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

fn write_to_buffer(
    field_value: &dyn Reflect,
    module: &naga::Module,
    ty: &naga::Type,
    buffer: &mut UniformBuffer<Vec<u8>>,
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
            let base = module
                .types
                .get_handle(*base)
                .expect("Array base type not found");
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

                if let Some(field) = reflect_struct.field(&name) {
                    let member_ty = module
                        .types
                        .get_handle(member.ty)
                        .expect("Struct member type not found");
                    write_to_buffer(field, module, &member_ty, buffer);
                } else {
                    panic!("Struct field not found: {:?}", member.name);
                }
            }
        }
        _ => {}
    }
}
