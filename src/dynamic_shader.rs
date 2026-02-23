use crate::module::Bindings;
use bevy::math::{Mat4, Vec2, Vec3, Vec4};
use bevy::reflect::utility::NonGenericTypeInfoCell;
use bevy::reflect::{
    ApplyError, FromReflect, FromType, GetTypeRegistration,
    PartialReflect, Reflect, ReflectFromPtr, ReflectKind, ReflectMut, ReflectOwned, ReflectRef,
    TypeInfo, TypePath, TypeRegistration, Typed,
};
use bevy::reflect::structs::{DynamicStruct, FieldIter, Struct, StructInfo};
use naga::{ArraySize, ScalarKind, StructMember, Type, TypeInner, VectorSize};
use std::any::Any;
use std::fmt::Formatter;

#[derive(TypePath, Debug, Default)]
pub struct DynamicShader {
    module: naga::Module,
    bindings: Bindings,
    storage: DynamicStruct,
}

impl FromReflect for DynamicShader {
    fn from_reflect(_reflect: &dyn PartialReflect) -> Option<DynamicShader> {
        None
    }
}

impl Clone for DynamicShader {
    fn clone(&self) -> Self {
        Self {
            module: self.module.clone(),
            bindings: self.bindings.clone(),
            storage: self.storage.to_dynamic_struct(),
        }
    }
}

impl DynamicShader {
    /// Creates a new DynamicShader with the given module
    pub fn new(module: naga::Module) -> Self {
        Self {
            module: module.clone(),
            bindings: Bindings::new(module),
            storage: DynamicStruct::default(),
        }
    }

    pub fn set_module(&mut self, module: naga::Module) {
        self.module = module.clone();
        self.bindings = Bindings::new(module);
    }

    /// Default initializes all fields in the shader
    pub fn init(&mut self) {
        let names: Vec<String> = self.bindings.names().map(|s| s.to_string()).collect();
        for name in names {
            self.field_mut(&name);
        }
    }

    /// Set the value of a field in the shader
    pub fn set<T: Reflect>(&mut self, name: &str, value: T) -> Result<(), ApplyError> {
        if let Some(field) = self.field_mut(name) {
            field.try_apply(value.as_partial_reflect())?;
        }

        Ok(())
    }

    /// Get the value of a field in the shader
    pub fn get<T: 'static>(&self, name: &str) -> Option<&T> {
        self.field(name)
            .and_then(|field| field.try_downcast_ref::<T>())
    }

    fn has_name(&self, name: &str) -> bool {
        self.bindings.has_name(name)
    }

    fn has_index(&self, index: usize) -> bool {
        self.bindings.get(index).is_some()
    }

    fn insert_default_field(&mut self, name: &str) {
        let Some(binding) = self.bindings.find_name(name) else {
            return;
        };

        self.insert_default_value(name, &binding.ty);
    }

    fn insert_default_value(&mut self, name: &str, ty: &Type) {
        match &ty.inner {
            TypeInner::Scalar(scalar) => match scalar.kind {
                ScalarKind::Sint => self.storage.insert(name, 0i32),
                ScalarKind::Uint => self.storage.insert(name, 0u32),
                ScalarKind::Float => self.storage.insert(name, 0.0f32),
                ScalarKind::Bool => self.storage.insert(name, 0u32),
                _ => {}
            },
            TypeInner::Vector { size, scalar } => match (size, scalar.kind) {
                (VectorSize::Bi, ScalarKind::Float) => self.storage.insert(name, Vec2::ZERO),
                (VectorSize::Tri, ScalarKind::Float) => self.storage.insert(name, Vec3::ZERO),
                (VectorSize::Quad, ScalarKind::Float) => self.storage.insert(name, Vec4::ZERO),
                _ => {}
            },
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => match (columns, rows, scalar.kind) {
                (VectorSize::Quad, VectorSize::Quad, ScalarKind::Float) => {
                    self.storage.insert(name, Mat4::IDENTITY)
                }
                _ => {}
            },
            TypeInner::Array { base, size, .. } => {
                self.insert_default_array(name, base, size);
            }
            TypeInner::Struct { members, .. } => {
                self.insert_default_struct(name, members);
            }
            _ => {}
        }
    }

    fn insert_default_array(
        &mut self,
        name: &str,
        base: &naga::Handle<Type>,
        array_size: &ArraySize,
    ) {
        let base_ty = &self.module.types[*base];

        match &base_ty.inner {
            TypeInner::Scalar(scalar) => match scalar.kind {
                ScalarKind::Sint => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![0i32; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                ScalarKind::Uint => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![0u32; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                ScalarKind::Float => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![0.0f32; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                ScalarKind::Bool => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![false; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                _ => {}
            },
            TypeInner::Vector { size, scalar } => match (size, scalar.kind) {
                (VectorSize::Bi, ScalarKind::Float) => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![Vec2::ZERO; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                (VectorSize::Tri, ScalarKind::Float) => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![Vec3::ZERO; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                (VectorSize::Quad, ScalarKind::Float) => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![Vec4::ZERO; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                _ => {}
            },
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => match (columns, rows, scalar.kind) {
                (VectorSize::Quad, VectorSize::Quad, ScalarKind::Float) => self.storage.insert(
                    name,
                    match array_size {
                        ArraySize::Constant(size) => vec![Mat4::IDENTITY; size.get() as usize],
                        ArraySize::Pending(_) | ArraySize::Dynamic => Vec::new(),
                    },
                ),
                _ => {}
            },
            TypeInner::Struct { .. } => {
                todo!("Array of struct types");
            }
            _ => {}
        }
    }

    fn insert_default_struct(&mut self, name: &str, members: &[StructMember]) {
        let mut default_struct = DynamicStruct::default();
        self.populate_default_struct(&mut default_struct, members);
        self.storage.insert(name, default_struct);
    }

    fn populate_default_struct(&self, struct_value: &mut DynamicStruct, members: &[StructMember]) {
        for member in members {
            let member_ty = &self.module.types[member.ty];
            let member_name = member.name.as_ref().expect("Struct member has no name");
            match &member_ty.inner {
                TypeInner::Scalar(scalar) => match scalar.kind {
                    ScalarKind::Sint => struct_value.insert(member_name, 0i32),
                    ScalarKind::Uint => struct_value.insert(member_name, 0u32),
                    ScalarKind::Float => struct_value.insert(member_name, 0.0f32),
                    ScalarKind::Bool => struct_value.insert(member_name, false),
                    _ => {}
                },
                TypeInner::Vector { size, scalar } => match (size, scalar.kind) {
                    (VectorSize::Bi, ScalarKind::Float) => {
                        struct_value.insert(member_name, Vec2::ZERO)
                    }
                    (VectorSize::Tri, ScalarKind::Float) => {
                        struct_value.insert(member_name, Vec3::ZERO)
                    }
                    (VectorSize::Quad, ScalarKind::Float) => {
                        struct_value.insert(member_name, Vec4::ZERO)
                    }
                    _ => {}
                },
                TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                } => match (columns, rows, scalar.kind) {
                    (VectorSize::Quad, VectorSize::Quad, ScalarKind::Float) => {
                        struct_value.insert(member_name, Mat4::IDENTITY)
                    }
                    _ => {}
                },
                _ => todo!("Struct member type: {:?}", member_ty),
            }
        }
    }
}

impl Typed for DynamicShader {
    fn type_info() -> &'static TypeInfo {
        static CELL: NonGenericTypeInfoCell = NonGenericTypeInfoCell::new();
        CELL.get_or_set(|| {
            let fields = [];
            let info = StructInfo::new::<Self>(&fields);
            TypeInfo::Struct(info)
        })
    }
}

impl GetTypeRegistration for DynamicShader {
    fn get_type_registration() -> TypeRegistration {
        let mut type_registration = TypeRegistration::of::<DynamicShader>();
        type_registration.insert::<ReflectFromPtr>(FromType::<DynamicShader>::from_type());
        type_registration
    }
}

impl Struct for DynamicShader {
    fn field(&self, name: &str) -> Option<&dyn PartialReflect> {
        self.bindings.find_name(name).and_then(|_binding| {
            self.storage.field(name)
        })
    }

    fn field_mut(&mut self, name: &str) -> Option<&mut dyn PartialReflect> {
        if !self.has_name(name) {
            return None;
        }

        if self.storage.field(name).is_none() {
            self.insert_default_field(name);
        }
        self.storage.field_mut(name)
    }

    fn field_at(&self, index: usize) -> Option<&dyn PartialReflect> {
        self.bindings.get(index)
            .and_then(|binding| binding.variable.name.as_ref().map(|s| s.as_str()))
            .and_then(|name| self.field(name))
    }

    fn field_at_mut(&mut self, index: usize) -> Option<&mut dyn PartialReflect> {
        if !self.has_index(index) {
            return None;
        }

        if self.storage.field_at(index).is_none() {
            let name = self.name_at(index).map(|s| s.to_string());
            if let Some(name) = name {
                self.insert_default_field(&name);
            }
        }
        self.storage.field_at_mut(index)
    }

    fn name_at(&self, index: usize) -> Option<&str> {
        self.bindings.get(index)
            .and_then(|binding| binding.name())
    }

    fn index_of_name(&self, name: &str) -> Option<usize> {
        self.bindings.iter().position(|b| b.name() == Some(name))
    }

    fn field_len(&self) -> usize {
        self.bindings.iter().count()
    }

    fn iter_fields(&self) -> FieldIter<'_> {
        FieldIter::new(self)
    }

    fn to_dynamic_struct(&self) -> DynamicStruct {
        self.storage.to_dynamic_struct()
    }
}

impl PartialReflect for DynamicShader {
    #[inline]
    fn get_represented_type_info(&self) -> Option<&'static TypeInfo> {
        Some(Self::type_info())
    }

    fn into_partial_reflect(self: Box<Self>) -> Box<dyn PartialReflect> {
        self
    }

    fn as_partial_reflect(&self) -> &dyn PartialReflect {
        self
    }

    fn as_partial_reflect_mut(&mut self) -> &mut dyn PartialReflect {
        self
    }

    fn try_into_reflect(self: Box<Self>) -> Result<Box<dyn Reflect>, Box<dyn PartialReflect>> {
        Ok(self)
    }

    fn try_as_reflect(&self) -> Option<&dyn Reflect> {
        Some(self)
    }

    fn try_as_reflect_mut(&mut self) -> Option<&mut dyn Reflect> {
        Some(self)
    }

    fn try_apply(&mut self, value: &dyn PartialReflect) -> Result<(), ApplyError> {
        if let ReflectRef::Struct(struct_value) = value.reflect_ref() {
            for (name, value) in struct_value.iter_fields() {
                if let Some(v) = self.field_mut(name) {
                    v.try_apply(value)?;
                }
            }
        } else {
            return Err(ApplyError::MismatchedKinds {
                from_kind: value.reflect_kind(),
                to_kind: ReflectKind::Struct,
            });
        }
        Ok(())
    }

    #[inline]
    fn reflect_kind(&self) -> ReflectKind {
        ReflectKind::Struct
    }

    #[inline]
    fn reflect_ref(&self) -> ReflectRef<'_> {
        ReflectRef::Struct(self)
    }

    #[inline]
    fn reflect_mut(&mut self) -> ReflectMut<'_> {
        ReflectMut::Struct(self)
    }

    #[inline]
    fn reflect_owned(self: Box<Self>) -> ReflectOwned {
        ReflectOwned::Struct(self)
    }

    fn reflect_partial_eq(&self, value: &dyn PartialReflect) -> Option<bool> {
        bevy::reflect::structs::struct_partial_eq(self, value)
    }

    fn debug(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.storage.debug(f)
    }
}

impl Reflect for DynamicShader {
    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_reflect(self: Box<Self>) -> Box<dyn Reflect> {
        self
    }

    fn as_reflect(&self) -> &dyn Reflect {
        self
    }

    fn as_reflect_mut(&mut self) -> &mut dyn Reflect {
        self
    }

    fn set(&mut self, value: Box<dyn Reflect>) -> Result<(), Box<dyn Reflect>> {
        let new: DynamicShader = value.take()?;
        self.storage = new.storage;
        Ok(())
    }
}
