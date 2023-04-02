// TODO

pub const Register = enum(u8) {
    // Address registers
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,

    // Data registers
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    d7,

    // Program counter
    pc,

    // condition code register
    ccr,

    /// TODO: make this the actual code used in machine code
    pub fn id(self: Register) u8 {
        return @enumToInt(self);
    }
};
