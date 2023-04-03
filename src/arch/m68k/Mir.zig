// TODO
const std = @import("std");
const Register = @import("bits.zig").Register;

instructions: std.MultiArrayList(Inst).Slice,
/// The meaning of this data is determined by `Inst.Tag` value.
extra: []const u32,

const Mir = @This();

pub const RegisterOrAddress = union {
    register: Register,
    address: u32,
};

pub const AddressMode = union {
    register: Register,
    address_in_register: Register,
    address_in_register_plus_offset: struct {
        register: Register,
        offset: u32,
    },
    address_in_register_postincrement: Register,
    address_in_register_predecrement: Register,
    immediate: u32,
    address: u32,
};

pub const Inst = struct {
    // TODO
    tag: Tag,
    data: Data,
    pub const Index = u32;
    pub const Tag = enum {
        // Yes, the capitalization is inconsistent.
        // It's the same way in official Motorola documenntation:
        // https://ia801904.us.archive.org/10/items/M68000PRM/M68000PRM.pdf
        //
        // Does NOT include 68010/68020/68030/68040/68060-specific instructions.
        // Also, does NOT include instructions that are only for privileged modes.

        // Add decimal with extend
        ABCD,
        // Add
        ADD,
        // Add address
        ADDA,
        /// Add immediate
        ///
        /// ADDI #imm, Dn
        /// or ADDI #imm, <address>
        ///
        /// Data is an immediate_and_register_or_address struct
        ADDI,
        // Add quick
        ADDQ,
        // Add extended
        ADDX,

        // And
        AND,
        /// And immediate
        ///
        /// ANDI #imm, Dn
        /// or ANDI #imm, <address>
        ///
        /// Data is an immediate_and_register_or_address.
        ANDI,
        // And immediate to CCR
        ANDI_to_CCR,

        // Arithmetic shift left
        ASL,
        // Arithmetic shift right
        ASR,

        // Branch conditionally
        Bcc,

        // Test a bit and change
        BCHG,
        // Test a bit and clear
        BCLR,

        // Branch
        BRA,

        // Test a bit and set
        BSET,

        // Branch to subroutine
        BSR,

        // Test a bit
        BTST,

        // Check register against bounds
        CHK,

        // Clear an operand
        CLR,

        // Compare
        CMP,
        // Compare address
        CMPA,
        // Compare immediate
        CMPI,
        // Compare memory
        CMPM,

        // Test condition, decrement, and branch
        DBcc,

        // Signed divide (WORD)
        DIVS,
        // Signed divide (LONG)
        DIVSL,
        // Unsigned divide (WORD)
        DIVU,
        // Unsigned divide (LONG)
        DIVUL,

        // exclusive-OR
        EOR,
        // exclusive-OR immediate
        EORI,
        // exclusive-OR immediate to CCR
        EORI_to_CCR,

        // exchange registers
        EXG,

        // sign-extend
        EXT,

        // Take illegal instruction trap
        ILLEGAL,

        // Jump
        JMP,
        // Jump to subroutine
        JSR,

        /// Load effective address
        ///
        /// LEA <ea>, An
        ///
        /// Data is a `register_and_address_mode` struct.
        LEA,

        /// Link and allocate
        ///
        /// LINK An, #<displacement>
        ///
        /// Data is a `register_and_displacement` struct.
        LINK,

        // Logical shift left
        LSL,
        // Logical shift right
        LSR,

        /// Move data from source to destination
        ///
        /// MOVE <source>, <destination>
        ///
        /// Data is a `src_dest`.
        MOVE,
        // Move address
        MOVEA,
        // Move to CCR
        MOVE_to_CCR,
        // Move multiple registers
        MOVEM,
        // Move peripheral data
        MOVEP,
        // Move quick
        MOVEQ,

        // Signed multiply
        MULS,
        // Unsigned multiply
        MULU,

        // Negate decimal with extend
        NBCD,
        // Negate
        NEG,
        // Negate with extend
        NEGX,

        /// No operation
        ///
        /// NOP
        ///
        /// Data is a `none`.
        NOP,


        // Logical complement
        NOT,

        // Inclusive-OR logical
        OR,
        // Inclusive-OR logical immediate
        ORI,
        // Inclusive-OR logical immediate to CCR
        ORI_to_CCR,

        // Push effective address
        PEA,

        // Rotate left without extennd
        ROL,
        // Rotate right without extend
        ROR,
        // Rotate left with extend
        ROXL,
        // Rotate right with extend
        ROXR,

        // Return and restore condition codes
        RTR,
        /// Return from subroutine
        ///
        /// RTS
        ///
        /// Data is a `none`.
        RTS,

        // Subtract decimal with extend
        SBCD,

        // Set according to condition
        Scc,

        // Subtract
        SUB,
        // Subtract address
        SUBA,
        // Subtract immediate
        SUBI,
        // Subtract quick
        SUBQ,
        // Subtract extended
        SUBX,

        // Swap register halves
        SWAP,

        // Test and set an operand
        TAS,

        // Trap
        TRAP,
        // Trap on overflow
        TRAPV,

        // Test an operand
        TST,

        /// Unlink
        ///
        /// UNLK An
        ///
        /// Data is a Register.
        UNLK,

        // Move from SR
        MOVE_from_SR,

        /// Pseudo-instruction: End of prologue
        dbg_prologue_end,
        /// Pseudo-instruction: Beginning of epilogue
        dbg_epilogue_begin,
        /// Pseudo-instruction: Update debug line
        dbg_line,
    };

    pub const Data = union {
        /// No data.
        none: void,

        /// A register.
        register: Register,

        /// References another Mir instruction.
        inst: Index,

        /// A register and a displacement value (1 word - ie an unsigned 16 bit integer)
        register_and_displacement: struct {
            register: Register,
            displacement: u16,
        },

        /// A register and an address mode.
        ///
        /// Used by e.g. LEA
        register_and_address_mode: struct {
            register: Register,
            address_mode: AddressMode,
        },

        /// Debug info: line and column
        ///
        /// Used by e.g. dbg_line
        dbg_line_column: struct {
            line: u32,
            column: u32,
        },

        /// Immediate value and a register or address
        ///
        /// Used by e.g. ADDI
        immediate_and_register_or_address: struct {
            immediate: u32,
            register_or_address: RegisterOrAddress,
        },

        /// Source and destination
        /// Used by e.g. MOVE
        src_dest: struct {
            src: AddressMode,
            dest: AddressMode,
        },
    };
};

pub fn deinit(mir: *Mir, gpa: std.mem.Allocator) void {
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    mir.* = undefined;
}
