use std::cmp::{max, min};

#[derive(Debug, Hash)]
pub struct Bitmap {
    bits: bitvec::vec::BitVec<u8>,
    width: usize,
    height: usize,
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
        for y in *top..min(*top + *height, self.height) {
            for x in *left..min(*left + *width, self.width) {
                if self.get(x, y) {
                    result.set(x, y, true);
                }
            }
        }
        result
    }
}
