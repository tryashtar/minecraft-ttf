use std::fmt::Display;

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum VanillaFontId {
    Default,
    Alt,
    Illageralt,
    Uniform,
}
impl Display for VanillaFontId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => write!(f, "default"),
            Self::Alt => write!(f, "alt"),
            Self::Illageralt => write!(f, "illageralt"),
            Self::Uniform => write!(f, "uniform"),
        }
    }
}
