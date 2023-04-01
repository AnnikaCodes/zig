// TODO

pub const Inst = struct {
    // TODO
    tag: Tag,
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
        // Add immediate
        ADDI,
        // Add quick
        ADDQ,
        // Add extended
        ADDX,

        // And
        AND,
        // And immediate
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

        // Load effective address
        LEA,

        // Link and allocate
        LINK,

        // Logical shift left
        LSL,
        // Logical shift right
        LSR,

        // Move data from source to destination
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

        // No operation
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
        // Return from subroutine
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

        // Unlink
        UNLK,

        // Move from SR
        MOVE_from_SR,
    };
};
