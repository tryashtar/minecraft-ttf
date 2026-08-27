use std::fmt::Display;

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
pub enum Style {
    Regular,
    Bold,
    Italic,
    BoldItalic,
}
impl Display for Style {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Regular => write!(f, "regular"),
            Self::Bold => write!(f, "bold"),
            Self::Italic => write!(f, "italic"),
            Self::BoldItalic => write!(f, "bold_italic"),
        }
    }
}
