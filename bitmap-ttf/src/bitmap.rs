use std::cmp::{max, min};

#[derive(Debug)]
pub struct Bitmap {
    bits: bitvec::vec::BitVec,
    width: usize,
    height: usize,
}

#[derive(Debug)]
pub struct Rectangle {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl Bitmap {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            bits: bitvec::bitvec![0; width * height],
            width,
            height,
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
        if left == self.width {
            return None;
        }
        Some(Rectangle {
            left,
            top,
            right,
            bottom,
        })
    }

    pub fn resized(&self, bounds: &Rectangle) -> Self {
        let Rectangle {
            left,
            top,
            right,
            bottom,
        } = bounds;
        let mut result = Bitmap::new(right - left, bottom - top);
        for y in *top..min(*bottom, self.height) {
            for x in *left..min(*right, self.width) {
                if self.get(x, y) {
                    result.set(x, y, true);
                }
            }
        }
        result
    }
}
