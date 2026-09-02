use std::{
    cmp::{max, min},
    fmt::Display,
};

#[derive(Debug, Hash)]
pub struct Bitmap {
    bits: bitvec::vec::BitVec<u8>,
    width: usize,
    height: usize,
}

impl Display for Bitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    write!(f, "█")?;
                } else {
                    write!(f, " ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Rectangle {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
}

impl Bitmap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            bits: bitvec::bitvec![u8, bitvec::order::Lsb0; 0; width * height],
            width,
            height,
        }
    }

    pub fn from_array(width: usize, height: usize, bits: bitvec::vec::BitVec<u8>) -> Option<Self> {
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
        if left == self.width || top == self.height {
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
        result.draw(self, *left, *top);
        result
    }

    pub fn draw(&mut self, other: &Bitmap, left: usize, top: usize) {
        for y in 0..min(other.height(), self.height - top) {
            for x in 0..min(other.width(), self.width - left) {
                if other.get(x, y) {
                    self.set(x + left, y + top, true);
                }
            }
        }
    }
}
