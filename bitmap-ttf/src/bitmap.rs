use std::{
    cmp::{max, min},
    fmt::Display,
};

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct Bitmap {
    bits: bitvec::vec::BitVec<u8, bitvec::order::Msb0>,
    width: usize,
    height: usize,
}

impl Display for Bitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "┌")?;
        write!(f, "{}", "─".repeat(self.width))?;
        writeln!(f, "┐")?;
        for y in 0..self.height {
            write!(f, "│")?;
            for x in 0..self.width {
                if self.get(x, y) {
                    write!(f, "█")?;
                } else {
                    write!(f, "░")?;
                }
            }
            writeln!(f, "│")?;
        }
        write!(f, "└")?;
        write!(f, "{}", "─".repeat(self.width))?;
        writeln!(f, "┘")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Rectangle {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl Bitmap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            bits: bitvec::bitvec![u8, bitvec::order::Msb0; 0; width * height],
            width,
            height,
        }
    }

    pub fn from_array(
        width: usize,
        height: usize,
        bits: bitvec::vec::BitVec<u8, bitvec::order::Msb0>,
    ) -> Option<Self> {
        if width * height == bits.len() {
            Some(Self {
                bits,
                width,
                height,
            })
        } else {
            None
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.bits[y * self.width + x]
    }

    pub fn try_get(&self, x: usize, y: usize) -> Option<bool> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.get(x, y))
    }

    pub fn set(&mut self, x: usize, y: usize, value: bool) {
        self.bits.set(y * self.width + x, value);
    }

    pub fn content_box(&self) -> Option<Rectangle> {
        let mut left = self.width;
        let mut top = self.height;
        let mut right = 0usize;
        let mut bottom = 0usize;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    left = min(left, x);
                    top = min(top, y);
                    right = max(right, x + 1);
                    bottom = max(bottom, y + 1);
                }
            }
        }
        if right == 0 {
            return None;
        }
        Some(Rectangle {
            left,
            top,
            width: right - left,
            height: bottom - top,
        })
    }

    pub fn resized(&self, bounds: &Rectangle) -> Self {
        let Rectangle {
            left,
            top,
            width,
            height,
        } = bounds;
        let mut result = Bitmap::new(*width, *height);
        self.draw_onto(&mut result, -(*left as isize), -(*top as isize));
        result
    }

    pub fn draw_onto(&self, destination: &mut Bitmap, left: isize, top: isize) {
        let src_left = (-left).max(0) as usize;
        let src_top = (-top).max(0) as usize;
        let dst_left = left.max(0) as usize;
        let dst_top = top.max(0) as usize;
        if src_left >= self.width
            || src_top >= self.height
            || dst_left >= destination.width
            || dst_top >= destination.height
        {
            return;
        }
        let width = (self.width - src_left).min(destination.width - dst_left);
        let height = (self.height - src_top).min(destination.height - dst_top);
        for y in 0..height {
            for x in 0..width {
                if self.get(src_left + x, src_top + y) {
                    destination.set(dst_left + x, dst_top + y, true);
                }
            }
        }
    }
}
