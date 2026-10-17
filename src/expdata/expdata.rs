use super::{ConstData, NormalData, ZeroData};
use ndarray::Array2;
use num_traits::Pow;
use pyo3::prelude::*;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

// A Datastruct which represents different kinds of experiment data (rely on "t")
//
// NormalData, ConstData, ZeroData, ErrData are the subtypes of ExpData
#[pyclass]
#[derive(Debug, Clone)]
pub enum ExpData {
    Normal { content: NormalData },
    Const { content: ConstData },
    Zero { content: ZeroData },
    Err {},
}

#[pymethods]
impl ExpData {
    #[inline]
    fn __str__(&self) -> String {
        format!("{}", self)
    }
    #[new]
    #[inline]
    fn __new__(arr: Vec<Vec<f64>>) -> ExpData {
        Array2::from_shape_vec(
            (arr.len(), arr[0].len()),
            arr.iter().flat_map(|x| x.iter()).cloned().collect(),
        )
        .unwrap()
        .into()
    }
    #[getter]
    #[inline]
    pub fn is_normal(&self) -> bool {
        match self {
            ExpData::Normal { content: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    pub fn is_const(&self) -> bool {
        match self {
            ExpData::Const { content: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    pub fn is_conserved(&self) -> bool {
        match self {
            ExpData::Const { content: _ } | ExpData::Zero { content: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            ExpData::Zero { content: _ } => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    pub fn is_err(&self) -> bool {
        match self {
            ExpData::Err {} => true,
            _ => false,
        }
    }
    #[getter]
    #[inline]
    fn n(&self) -> usize {
        match self {
            ExpData::Normal { content } => content.n,
            ExpData::Const { content: _ } => panic!("ConstData has no n"),
            ExpData::Zero { content: _ } => panic!("ZeroData has no n"),
            ExpData::Err {} => panic!("ErrData has no n"),
        }
    }
    #[getter]
    #[inline]
    fn repeat_time(&self) -> usize {
        match self {
            ExpData::Normal { content } => content.repeat_time,
            ExpData::Const { content: _ } => panic!("ConstData has no repeat_time"),
            ExpData::Zero { content: _ } => panic!("ZeroData has no repeat_time"),
            ExpData::Err {} => panic!("ErrData has no repeat_time"),
        }
    }
    #[getter]
    #[inline]
    fn get_normal_data(&self) -> NormalData {
        self.unwrap_normal_data().clone()
    }
    #[getter]
    #[inline]
    fn get_const_data(&self) -> ConstData {
        self.unwrap_const_data().clone()
    }
    #[inline]
    fn get_normal_form(&self, n: usize, repeat_time: usize) -> NormalData {
        self.to_normal_data(n, repeat_time)
    }
    #[staticmethod]
    #[inline]
    fn from_elem(mean: f64, std: f64, n: usize, repeat_time: usize) -> PyResult<ExpData> {
        Ok(ExpData::Normal {
            content: NormalData::from_elem(mean, std, n, repeat_time),
        })
    }
    #[inline]
    #[staticmethod]
    fn from_normal_data(content: NormalData) -> PyResult<ExpData> {
        Ok(content.into())
    }
    #[inline]
    #[staticmethod]
    pub fn from_const_data(content: ConstData) -> PyResult<ExpData> {
        Ok(content.into())
    }
    #[inline]
    #[staticmethod]
    pub fn from_const(mean: f64, std: f64) -> PyResult<ExpData> {
        Ok(ConstData::new(mean, std).into())
    }
    #[inline]
    #[getter]
    pub fn force_to_const_data(&self) -> Option<ConstData> {
        match self {
            ExpData::Normal { content } => Some(content.to_const_data()),
            ExpData::Const { content } => Some(content.clone()),
            _ => None,
        }
    }
    #[inline]
    #[getter]
    pub fn calc_mean(&self) -> Option<ConstData> {
        match self {
            ExpData::Normal { content } => content.calc_mean(),
            ExpData::Const { content } => Some(content.clone()),
            _ => None,
        }
    }
    #[inline]
    fn __add__(&self, other: &ExpData) -> PyResult<ExpData> {
        Ok(self + other)
    }
    #[inline]
    fn __sub__(&self, other: &ExpData) -> PyResult<ExpData> {
        Ok(self - other)
    }
    #[inline]
    fn __mul__(&self, other: &ExpData) -> PyResult<ExpData> {
        Ok(self * other)
    }
    #[inline]
    fn __rmul__(&self, other: f64) -> PyResult<ExpData> {
        Ok(self * &ExpData::from(other))
    }
    #[inline]
    fn __truediv__(&self, other: &ExpData) -> PyResult<ExpData> {
        Ok(self / other)
    }
    #[inline]
    fn __neg__(&self) -> PyResult<ExpData> {
        Ok(-self.clone())
    }
    #[inline]
    fn __powi__(&self, other: i32) -> PyResult<ExpData> {
        Ok(self.powi(other))
    }
    #[inline]
    fn __powq__(&self, numerator: i64, denominator: i64) -> PyResult<ExpData> {
        Ok(self.pow_rational(numerator, denominator))
    }
    #[inline]
    fn __diff__(&self, other: &ExpData) -> PyResult<ExpData> {
        Ok(self.diff(other))
    }
    #[inline]
    fn __difftau__(&self) -> PyResult<ExpData> {
        Ok(self.diff_tau())
    }
    #[staticmethod]
    #[inline]
    pub fn wrapped_list_of_const_data(
        list_constdata: Vec<Option<ConstData>>,
        repeat_time: usize,
    ) -> ExpData {
        NormalData::wrapped_list_of_const_data(list_constdata, repeat_time).into()
    }
}

impl ExpData {
    #[inline]
    pub fn unwrap_normal_data(&self) -> &NormalData {
        match self {
            ExpData::Normal { content } => content,
            _ => panic!("unwrap_normal_data called on non-NormalData"),
        }
    }
    #[inline]
    pub fn unwrap_const_data(&self) -> &ConstData {
        match self {
            ExpData::Const { content } => content,
            _ => panic!("unwrap_const_data called on non-ConstData"),
        }
    }
    #[inline]
    pub fn unwrap_zero_data(&self) -> &ZeroData {
        match self {
            ExpData::Zero { content } => content,
            _ => panic!("unwrap_zero_data called on non-ZeroData"),
        }
    }
    #[inline]
    pub fn to_const_data(&self) -> Option<ConstData> {
        match self {
            ExpData::Const { content } => Some(content.clone()),
            _ => None,
        }
    }
    #[inline]
    pub fn from_arr2(arr: Array2<f64>) -> ExpData {
        arr.into()
    }
    #[inline]
    pub fn from_exact_const(value: i32) -> ExpData {
        ConstData::from(value).into()
    }
    #[inline]
    pub fn to_normal_data(&self, n: usize, repeat_time: usize) -> NormalData {
        match self {
            ExpData::Normal { content } => {
                assert_eq!(n, content.n);
                assert_eq!(repeat_time, content.repeat_time);
                content.clone()
            }
            ExpData::Const { content } => NormalData::from_const_data(content, n, repeat_time),
            ExpData::Zero { content } => NormalData::from_zero_data(content, n, repeat_time),
            ExpData::Err {} => panic!("Cannot convert ErrData to NormalData"),
        }
    }
}

impl fmt::Display for ExpData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ExpData::Normal { content } => write!(f, "{}", content),
            ExpData::Const { content } => write!(f, "{}", content),
            ExpData::Zero { content } => write!(f, "{}", content),
            ExpData::Err {} => write!(f, "ExpData.Err"),
        }
    }
}

//
// Operator overloading
//

impl Pow<ExpData> for ExpData {
    type Output = ExpData;
    #[inline]
    fn pow(self, other: ExpData) -> ExpData {
        match other {
            ExpData::Const { content } => {
                match content {
                    ConstData::Exact { value } => self.powi(value),
                    ConstData::Data { mean, std } => self.powf(mean, std),
                }
            }
            ExpData::Zero { content } => {
                if content.err == 0. {
                    ExpData::from_exact_const(1)
                } else {
                    ExpData::Err {}
                    // TODO
                }
            }
            _ => ExpData::Err {},
        }
    }
}

pub trait Diff<Rhs = Self> {
    type Output;
    fn diff(&self, other: Rhs) -> Self::Output;
    fn diff_n(&self, other: Rhs, n: usize) -> Self::Output;
}

// implement the Add trait for ExpData
impl Add<&ZeroData> for &ExpData {
    type Output = ExpData;
    #[inline]
    fn add(self, rhs: &ZeroData) -> Self::Output {
        match self {
            ExpData::Err {} => ExpData::Err {},
            ExpData::Zero { content } => (content + rhs).into(),
            ExpData::Const { content } => (content + rhs).into(),
            ExpData::Normal { content } => {
                let rhs = NormalData::from_zero_data(rhs, self.n(), self.repeat_time());
                (content + &rhs).into()
            }
        }
    }
}
impl Add for ExpData {
    type Output = ExpData;
    #[inline]
    fn add(self, other: ExpData) -> ExpData {
        (&self) + (&other)
    }
}
impl Add for &ExpData {
    type Output = ExpData;
    #[inline]
    fn add(self, other: &ExpData) -> ExpData {
        if self.is_err() || other.is_err() {
            ExpData::Err {}
        } else if self.is_zero() {
            other + self.unwrap_zero_data()
        } else if other.is_zero() {
            self + other.unwrap_zero_data()
        } else if self.is_const() && other.is_const() {
            (self.unwrap_const_data() + other.unwrap_const_data()).into()
        } else {
            let n = if !self.is_const() {
                self.n()
            } else {
                other.n()
            };
            let repeat_time = if !self.is_const() {
                self.repeat_time()
            } else {
                other.repeat_time()
            };
            (self.to_normal_data(n, repeat_time) + other.to_normal_data(n, repeat_time)).into()
        }
    }
}

// implement the Sub trait for ExpData
impl Sub for ExpData {
    type Output = ExpData;
    #[inline]
    fn sub(self, other: ExpData) -> ExpData {
        (&self) - (&other)
    }
}
impl Sub<&ZeroData> for &ExpData {
    type Output = ExpData;
    #[inline]
    fn sub(self, other: &ZeroData) -> ExpData {
        match self {
            ExpData::Err {} => ExpData::Err {},
            ExpData::Zero { content } => (content - other).into(),
            ExpData::Const { content } => (content - other).into(),
            ExpData::Normal { content } => {
                let rhs = NormalData::from_zero_data(other, self.n(), self.repeat_time());
                (content - &rhs).into()
            }
        }
    }
}
impl Sub for &ExpData {
    type Output = ExpData;
    #[inline]
    fn sub(self, other: &ExpData) -> ExpData {
        if self.is_err() || other.is_err() {
            return ExpData::Err {};
        }
        if self.is_zero() {
            return -other;
        }
        if other.is_zero() {
            return self.clone();
        }
        if self.is_const() && other.is_const() {
            (self.unwrap_const_data() - other.unwrap_const_data()).into()
        } else {
            let n = if !self.is_const() {
                self.n()
            } else {
                other.n()
            };
            let repeat_time = if !self.is_const() {
                self.repeat_time()
            } else {
                other.repeat_time()
            };
            (self.to_normal_data(n, repeat_time) - other.to_normal_data(n, repeat_time)).into()
        }
    }
}

// implement the Mul trait for ExpData
impl Mul for ExpData {
    type Output = ExpData;
    #[inline]
    fn mul(self, other: ExpData) -> ExpData {
        (&self) * (&other)
    }
}

impl Mul<&ZeroData> for &ExpData {
    type Output = ExpData;
    #[inline]
    fn mul(self, other: &ZeroData) -> ExpData {
        match self {
            ExpData::Err {} => ExpData::Err {},
            ExpData::Zero { content } => (content * other).into(),
            _ => self
                .calc_mean()
                .and_then(|x| Some((other * x.mean()).into()))
                .unwrap_or(ExpData::Err {}),
        }
    }
}

impl Mul for &ExpData {
    type Output = ExpData;
    #[inline]
    fn mul(self, other: &ExpData) -> ExpData {
        if self.is_err() || other.is_err() {
            ExpData::Err {}
        } else if self.is_zero() {
            other * self.unwrap_zero_data()
        } else if other.is_zero() {
            self * other.unwrap_zero_data()
        } else if self.is_const() && other.is_const() {
            (self.unwrap_const_data() * other.unwrap_const_data()).into()
        } else {
            let n = if !self.is_const() {
                self.n()
            } else {
                other.n()
            };
            let repeat_time = if !self.is_const() {
                self.repeat_time()
            } else {
                other.repeat_time()
            };
            (self.to_normal_data(n, repeat_time) * other.to_normal_data(n, repeat_time)).into()
        }
    }
}

impl Div<&ExpData> for &ZeroData {
    type Output = ExpData;
    #[inline]
    fn div(self, other: &ExpData) -> ExpData {
        other
            .calc_mean()
            .and_then(|x| match x.mean().abs() {
                0. => Some(ExpData::Err {}),
                x => Some((self / x).into()),
            })
            .unwrap_or(ExpData::Err {})
    }
}

// implement the Div trait for ExpData
impl Div for ExpData {
    type Output = ExpData;
    #[inline]
    fn div(self, other: ExpData) -> ExpData {
        (&self) / (&other)
    }
}
impl Div for &ExpData {
    type Output = ExpData;
    #[inline]
    fn div(self, other: &ExpData) -> ExpData {
        if self.is_err() || other.is_err() || other.is_zero() {
            ExpData::Err {}
        } else if self.is_zero() {
            self.unwrap_zero_data() / other
        } else if self.is_const() && other.is_const() {
            (self.unwrap_const_data() / other.unwrap_const_data()).into()
        } else {
            let n = if !self.is_const() {
                self.n()
            } else {
                other.n()
            };
            let repeat_time = if !self.is_const() {
                self.repeat_time()
            } else {
                other.repeat_time()
            };
            (self.to_normal_data(n, repeat_time) / other.to_normal_data(n, repeat_time)).into()
        }
    }
}

// implement the AddAssign trait for ExpData
impl AddAssign for ExpData {
    #[inline]
    fn add_assign(&mut self, other: ExpData) {
        *self = &*self + &other;
    }
}

impl Neg for ExpData {
    type Output = ExpData;
    #[inline]
    fn neg(self) -> ExpData {
        -&self
    }
}

impl Neg for &ExpData {
    type Output = ExpData;
    #[inline]
    fn neg(self) -> ExpData {
        match self {
            ExpData::Normal { content } => (-content).into(),
            ExpData::Const { content } => (-content).into(),
            ExpData::Zero { content } => content.clone().into(),
            ExpData::Err {} => ExpData::Err {},
        }
    }
}

impl ExpData {
    #[inline]
    pub fn powi(&self, other: i32) -> ExpData {
        match self {
            ExpData::Normal { content } => content.pow(other).into(),
            ExpData::Const { content } => content.pow(other).into(),
            ExpData::Zero { content } => content.pow(other).into(),
            ExpData::Err {} => ExpData::Err {},
        }
    }
    /// Raise data to a real constant power.  The second argument is the
    /// uncertainty of the exponent and is propagated for constant bases.
    pub fn powf(&self, exponent: f64, exponent_std: f64) -> ExpData {
        match self {
            ExpData::Normal { content } => {
                content.map_values(|value| value.powf(exponent)).into()
            }
            ExpData::Const { content } => {
                let base = content.mean();
                let value = base.powf(exponent);
                if !value.is_finite() {
                    return ExpData::Err {};
                }
                let base_error = if base == 0.0 {
                    0.0
                } else {
                    (exponent * base.powf(exponent - 1.0) * content.std()).abs()
                };
                let exponent_error = if base > 0.0 {
                    (value * base.ln() * exponent_std).abs()
                } else {
                    0.0
                };
                ConstData::new(
                    value,
                    (base_error.powi(2) + exponent_error.powi(2)).sqrt(),
                ).into()
            }
            ExpData::Zero { content } => {
                if exponent > 0.0 {
                    super::ZeroData::new(content.err.powf(exponent)).into()
                } else if exponent == 0.0 {
                    ExpData::from_exact_const(1)
                } else {
                    ExpData::Err {}
                }
            }
            ExpData::Err {} => ExpData::Err {},
        }
    }

    /// Exact rational-power evaluation, including real odd roots of negative
    /// observations.  Even roots retain the standard real-domain restriction.
    pub fn pow_rational(&self, numerator: i64, denominator: i64) -> ExpData {
        if denominator == 0 {
            return ExpData::Err {};
        }
        let denominator = denominator.abs();
        let exponent = numerator as f64 / denominator as f64;
        let real_power = move |value: f64| {
            if value >= 0.0 {
                value.powf(exponent)
            } else if denominator % 2 == 1 {
                let magnitude = (-value).powf(exponent);
                if numerator.abs() % 2 == 1 { -magnitude } else { magnitude }
            } else {
                f64::NAN
            }
        };
        match self {
            ExpData::Normal { content } => content.map_values(real_power).into(),
            ExpData::Const { content } => {
                let value = real_power(content.mean());
                if !value.is_finite() {
                    return ExpData::Err {};
                }
                let derivative = if content.mean() == 0.0 {
                    0.0
                } else {
                    exponent * real_power(content.mean()) / content.mean()
                };
                ConstData::new(value, derivative.abs() * content.std()).into()
            }
            ExpData::Zero { content } => {
                if numerator > 0 {
                    super::ZeroData::new(content.err.powf(exponent)).into()
                } else if numerator == 0 {
                    ExpData::from_exact_const(1)
                } else {
                    ExpData::Err {}
                }
            }
            ExpData::Err {} => ExpData::Err {},
        }
    }
    fn map_transcendental<F, D>(&self, function: F, derivative: D) -> ExpData
    where
        F: Fn(f64) -> f64 + Copy,
        D: Fn(f64) -> f64 + Copy,
    {
        match self {
            ExpData::Normal { content } => content.map_values(function).into(),
            ExpData::Const { content } => ConstData::new(
                function(content.mean()),
                derivative(content.mean()).abs() * content.std(),
            ).into(),
            ExpData::Zero { content } => ConstData::new(
                function(0.0),
                derivative(0.0).abs() * content.err,
            ).into(),
            ExpData::Err {} => ExpData::Err {},
        }
    }
    #[inline]
    pub fn sin(&self) -> ExpData {
        self.map_transcendental(f64::sin, f64::cos)
    }
    #[inline]
    pub fn cos(&self) -> ExpData {
        self.map_transcendental(f64::cos, |x| -x.sin())
    }
    #[inline]
    pub fn exp(&self) -> ExpData {
        self.map_transcendental(f64::exp, f64::exp)
    }
    #[inline]
    pub fn log(&self) -> ExpData {
        self.map_transcendental(f64::ln, |x| 1.0 / x)
    }
    #[inline]
    pub fn diff_tau(&self) -> ExpData {
        match self {
            ExpData::Normal { content } => content.diff_tau().into(),
            ExpData::Const { content } => {
                // TODO! analyze the error
                ZeroData::new(content.std()).into()
            }
            ExpData::Zero { content } => {
                // TODO! analyze the error
                ZeroData::new(content.err).into()
            }
            ExpData::Err {} => ExpData::Err {},
        }
    }
}

impl Diff for &ExpData {
    type Output = ExpData;
    #[inline]
    fn diff(&self, other: &ExpData) -> ExpData {
        if self.is_err() || other.is_err() {
            return ExpData::Err {};
        } else if self.is_zero() {
            // NOTE! todo: analyze the error
            ZeroData::new(self.unwrap_zero_data().err).into()
        } else if self.is_const() {
            // NOTE! todo: analyze the error
            ZeroData::new(self.unwrap_const_data().std()).into()
        } else if other.is_zero() || other.is_const() {
            ExpData::Err {}
        } else {
            if other.unwrap_normal_data().is_conserved_piecewise() {
                ExpData::Err {}
            } else {
                (self.unwrap_normal_data().diff_tau() / other.unwrap_normal_data().diff_tau())
                    .into()
            }
        }
    }
    #[inline]
    fn diff_n(&self, other: &ExpData, n: usize) -> ExpData {
        assert!(n > 0 && n < 5);
        if n == 1 {
            self.diff(other)
        } else {
            (&self.diff(other)).diff_n(other, n - 1)
        }
    }
}
