//! Semantic analysis of ZIR instructions.
//! Shared to every Block. Stored on the stack.
//! State used for compiling a ZIR into AIR.
//! Transforms untyped ZIR instructions into semantically-analyzed AIR instructions.
//! Does type checking, comptime control flow, and safety-check generation.
//! This is the the heart of the Zig compiler.

mod: *Module,
/// Alias to `mod.gpa`.
gpa: Allocator,
/// Points to the temporary arena allocator of the Sema.
/// This arena will be cleared when the sema is destroyed.
arena: Allocator,
/// Points to the arena allocator for the owner_decl.
/// This arena will persist until the decl is invalidated.
perm_arena: Allocator,
code: Zir,
air_instructions: std.MultiArrayList(Air.Inst) = .{},
air_extra: std.ArrayListUnmanaged(u32) = .{},
air_values: std.ArrayListUnmanaged(Value) = .{},
/// Maps ZIR to AIR.
inst_map: InstMap = .{},
/// When analyzing an inline function call, owner_decl is the Decl of the caller
/// and `src_decl` of `Block` is the `Decl` of the callee.
/// This `Decl` owns the arena memory of this `Sema`.
owner_decl: *Decl,
/// For an inline or comptime function call, this will be the root parent function
/// which contains the callsite. Corresponds to `owner_decl`.
owner_func: ?*Module.Fn,
/// The function this ZIR code is the body of, according to the source code.
/// This starts out the same as `owner_func` and then diverges in the case of
/// an inline or comptime function call.
func: ?*Module.Fn,
/// When semantic analysis needs to know the return type of the function whose body
/// is being analyzed, this `Type` should be used instead of going through `func`.
/// This will correctly handle the case of a comptime/inline function call of a
/// generic function which uses a type expression for the return type.
/// The type will be `void` in the case that `func` is `null`.
fn_ret_ty: Type,
branch_quota: u32 = 1000,
branch_count: u32 = 0,
/// Populated when returning `error.ComptimeBreak`. Used to communicate the
/// break instruction up the stack to find the corresponding Block.
comptime_break_inst: Zir.Inst.Index = undefined,
/// This field is updated when a new source location becomes active, so that
/// instructions which do not have explicitly mapped source locations still have
/// access to the source location set by the previous instruction which did
/// contain a mapped source location.
src: LazySrcLoc = .{ .token_offset = 0 },
decl_val_table: std.AutoHashMapUnmanaged(*Decl, Air.Inst.Ref) = .{},
/// When doing a generic function instantiation, this array collects a
/// `Value` object for each parameter that is comptime known and thus elided
/// from the generated function. This memory is allocated by a parent `Sema` and
/// owned by the values arena of the Sema owner_decl.
comptime_args: []TypedValue = &.{},
/// Marks the function instruction that `comptime_args` applies to so that we
/// don't accidentally apply it to a function prototype which is used in the
/// type expression of a generic function parameter.
comptime_args_fn_inst: Zir.Inst.Index = 0,
/// When `comptime_args` is provided, this field is also provided. It was used as
/// the key in the `monomorphed_funcs` set. The `func` instruction is supposed
/// to use this instead of allocating a fresh one. This avoids an unnecessary
/// extra hash table lookup in the `monomorphed_funcs` set.
/// Sema will set this to null when it takes ownership.
preallocated_new_func: ?*Module.Fn = null,
/// The key is `constant` AIR instructions to types that must be fully resolved
/// after the current function body analysis is done.
/// TODO: after upgrading to use InternPool change the key here to be an
/// InternPool value index.
types_to_resolve: std.ArrayListUnmanaged(Air.Inst.Ref) = .{},

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const log = std.log.scoped(.sema);

const Sema = @This();
const Value = @import("value.zig").Value;
const Type = @import("type.zig").Type;
const TypedValue = @import("TypedValue.zig");
const Air = @import("Air.zig");
const Zir = @import("Zir.zig");
const Module = @import("Module.zig");
const trace = @import("tracy.zig").trace;
const Namespace = Module.Namespace;
const CompileError = Module.CompileError;
const SemaError = Module.SemaError;
const Decl = Module.Decl;
const CaptureScope = Module.CaptureScope;
const WipCaptureScope = Module.WipCaptureScope;
const LazySrcLoc = Module.LazySrcLoc;
const RangeSet = @import("RangeSet.zig");
const target_util = @import("target.zig");
const Package = @import("Package.zig");
const crash_report = @import("crash_report.zig");
const build_options = @import("build_options");

pub const InstMap = std.AutoHashMapUnmanaged(Zir.Inst.Index, Air.Inst.Ref);

/// This is the context needed to semantically analyze ZIR instructions and
/// produce AIR instructions.
/// This is a temporary structure stored on the stack; references to it are valid only
/// during semantic analysis of the block.
pub const Block = struct {
    parent: ?*Block,
    /// Shared among all child blocks.
    sema: *Sema,
    /// This Decl is the Decl according to the Zig source code corresponding to this Block.
    /// This can vary during inline or comptime function calls. See `Sema.owner_decl`
    /// for the one that will be the same for all Block instances.
    src_decl: *Decl,
    /// The namespace to use for lookups from this source block
    /// When analyzing fields, this is different from src_decl.src_namepsace.
    namespace: *Namespace,
    /// The AIR instructions generated for this block.
    instructions: std.ArrayListUnmanaged(Air.Inst.Index),
    // `param` instructions are collected here to be used by the `func` instruction.
    params: std.ArrayListUnmanaged(Param) = .{},

    wip_capture_scope: *CaptureScope,

    label: ?*Label = null,
    inlining: ?*Inlining,
    /// If runtime_index is not 0 then one of these is guaranteed to be non null.
    runtime_cond: ?LazySrcLoc = null,
    runtime_loop: ?LazySrcLoc = null,
    /// Non zero if a non-inline loop or a runtime conditional have been encountered.
    /// Stores to to comptime variables are only allowed when var.runtime_index <= runtime_index.
    runtime_index: u32 = 0,

    is_comptime: bool,
    is_typeof: bool = false,

    /// when null, it is determined by build mode, changed by @setRuntimeSafety
    want_safety: ?bool = null,

    c_import_buf: ?*std.ArrayList(u8) = null,

    /// type of `err` in `else => |err|`
    switch_else_err_ty: ?Type = null,

    const Param = struct {
        /// `noreturn` means `anytype`.
        ty: Type,
        is_comptime: bool,
        name: []const u8,
    };

    /// This `Block` maps a block ZIR instruction to the corresponding
    /// AIR instruction for break instruction analysis.
    pub const Label = struct {
        zir_block: Zir.Inst.Index,
        merges: Merges,
    };

    /// This `Block` indicates that an inline function call is happening
    /// and return instructions should be analyzed as a break instruction
    /// to this AIR block instruction.
    /// It is shared among all the blocks in an inline or comptime called
    /// function.
    pub const Inlining = struct {
        comptime_result: Air.Inst.Ref,
        merges: Merges,
        err: ?*Module.ErrorMsg = null,
    };

    pub const Merges = struct {
        block_inst: Air.Inst.Index,
        /// Separate array list from break_inst_list so that it can be passed directly
        /// to resolvePeerTypes.
        results: std.ArrayListUnmanaged(Air.Inst.Ref),
        /// Keeps track of the break instructions so that the operand can be replaced
        /// if we need to add type coercion at the end of block analysis.
        /// Same indexes, capacity, length as `results`.
        br_list: std.ArrayListUnmanaged(Air.Inst.Index),
    };

    /// For debugging purposes.
    pub fn dump(block: *Block, mod: Module) void {
        Zir.dumpBlock(mod, block);
    }

    pub fn makeSubBlock(parent: *Block) Block {
        return .{
            .parent = parent,
            .sema = parent.sema,
            .src_decl = parent.src_decl,
            .namespace = parent.namespace,
            .instructions = .{},
            .wip_capture_scope = parent.wip_capture_scope,
            .label = null,
            .inlining = parent.inlining,
            .is_comptime = parent.is_comptime,
            .is_typeof = parent.is_typeof,
            .runtime_cond = parent.runtime_cond,
            .runtime_loop = parent.runtime_loop,
            .runtime_index = parent.runtime_index,
            .want_safety = parent.want_safety,
            .c_import_buf = parent.c_import_buf,
            .switch_else_err_ty = parent.switch_else_err_ty,
        };
    }

    pub fn wantSafety(block: *const Block) bool {
        return block.want_safety orelse switch (block.sema.mod.optimizeMode()) {
            .Debug => true,
            .ReleaseSafe => true,
            .ReleaseFast => false,
            .ReleaseSmall => false,
        };
    }

    pub fn getFileScope(block: *Block) *Module.File {
        return block.namespace.file_scope;
    }

    fn addTy(
        block: *Block,
        tag: Air.Inst.Tag,
        ty: Type,
    ) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = tag,
            .data = .{ .ty = ty },
        });
    }

    fn addTyOp(
        block: *Block,
        tag: Air.Inst.Tag,
        ty: Type,
        operand: Air.Inst.Ref,
    ) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = tag,
            .data = .{ .ty_op = .{
                .ty = try block.sema.addType(ty),
                .operand = operand,
            } },
        });
    }

    fn addBitCast(block: *Block, ty: Type, operand: Air.Inst.Ref) Allocator.Error!Air.Inst.Ref {
        return block.addInst(.{
            .tag = .bitcast,
            .data = .{ .ty_op = .{
                .ty = try block.sema.addType(ty),
                .operand = operand,
            } },
        });
    }

    fn addNoOp(block: *Block, tag: Air.Inst.Tag) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = tag,
            .data = .{ .no_op = {} },
        });
    }

    fn addUnOp(
        block: *Block,
        tag: Air.Inst.Tag,
        operand: Air.Inst.Ref,
    ) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = tag,
            .data = .{ .un_op = operand },
        });
    }

    fn addBr(
        block: *Block,
        target_block: Air.Inst.Index,
        operand: Air.Inst.Ref,
    ) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = .br,
            .data = .{ .br = .{
                .block_inst = target_block,
                .operand = operand,
            } },
        });
    }

    fn addBinOp(
        block: *Block,
        tag: Air.Inst.Tag,
        lhs: Air.Inst.Ref,
        rhs: Air.Inst.Ref,
    ) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = tag,
            .data = .{ .bin_op = .{
                .lhs = lhs,
                .rhs = rhs,
            } },
        });
    }

    fn addArg(block: *Block, ty: Type) error{OutOfMemory}!Air.Inst.Ref {
        return block.addInst(.{
            .tag = .arg,
            .data = .{ .ty = ty },
        });
    }

    fn addStructFieldPtr(
        block: *Block,
        struct_ptr: Air.Inst.Ref,
        field_index: u32,
        ptr_field_ty: Type,
    ) !Air.Inst.Ref {
        const ty = try block.sema.addType(ptr_field_ty);
        const tag: Air.Inst.Tag = switch (field_index) {
            0 => .struct_field_ptr_index_0,
            1 => .struct_field_ptr_index_1,
            2 => .struct_field_ptr_index_2,
            3 => .struct_field_ptr_index_3,
            else => {
                return block.addInst(.{
                    .tag = .struct_field_ptr,
                    .data = .{ .ty_pl = .{
                        .ty = ty,
                        .payload = try block.sema.addExtra(Air.StructField{
                            .struct_operand = struct_ptr,
                            .field_index = field_index,
                        }),
                    } },
                });
            },
        };
        return block.addInst(.{
            .tag = tag,
            .data = .{ .ty_op = .{
                .ty = ty,
                .operand = struct_ptr,
            } },
        });
    }

    fn addStructFieldVal(
        block: *Block,
        struct_val: Air.Inst.Ref,
        field_index: u32,
        field_ty: Type,
    ) !Air.Inst.Ref {
        return block.addInst(.{
            .tag = .struct_field_val,
            .data = .{ .ty_pl = .{
                .ty = try block.sema.addType(field_ty),
                .payload = try block.sema.addExtra(Air.StructField{
                    .struct_operand = struct_val,
                    .field_index = field_index,
                }),
            } },
        });
    }

    fn addSliceElemPtr(
        block: *Block,
        slice: Air.Inst.Ref,
        elem_index: Air.Inst.Ref,
        elem_ptr_ty: Type,
    ) !Air.Inst.Ref {
        return block.addInst(.{
            .tag = .slice_elem_ptr,
            .data = .{ .ty_pl = .{
                .ty = try block.sema.addType(elem_ptr_ty),
                .payload = try block.sema.addExtra(Air.Bin{
                    .lhs = slice,
                    .rhs = elem_index,
                }),
            } },
        });
    }

    fn addPtrElemPtr(
        block: *Block,
        array_ptr: Air.Inst.Ref,
        elem_index: Air.Inst.Ref,
        elem_ptr_ty: Type,
    ) !Air.Inst.Ref {
        const ty_ref = try block.sema.addType(elem_ptr_ty);
        return block.addPtrElemPtrTypeRef(array_ptr, elem_index, ty_ref);
    }

    fn addPtrElemPtrTypeRef(
        block: *Block,
        array_ptr: Air.Inst.Ref,
        elem_index: Air.Inst.Ref,
        elem_ptr_ty: Air.Inst.Ref,
    ) !Air.Inst.Ref {
        return block.addInst(.{
            .tag = .ptr_elem_ptr,
            .data = .{ .ty_pl = .{
                .ty = elem_ptr_ty,
                .payload = try block.sema.addExtra(Air.Bin{
                    .lhs = array_ptr,
                    .rhs = elem_index,
                }),
            } },
        });
    }

    fn addCmpVector(block: *Block, lhs: Air.Inst.Ref, rhs: Air.Inst.Ref, cmp_op: std.math.CompareOperator, vector_ty: Air.Inst.Ref) !Air.Inst.Ref {
        return block.addInst(.{
            .tag = .cmp_vector,
            .data = .{ .ty_pl = .{
                .ty = vector_ty,
                .payload = try block.sema.addExtra(Air.VectorCmp{
                    .lhs = lhs,
                    .rhs = rhs,
                    .op = Air.VectorCmp.encodeOp(cmp_op),
                }),
            } },
        });
    }

    fn addAggregateInit(
        block: *Block,
        aggregate_ty: Type,
        elements: []const Air.Inst.Ref,
    ) !Air.Inst.Ref {
        const sema = block.sema;
        const ty_ref = try sema.addType(aggregate_ty);
        try sema.air_extra.ensureUnusedCapacity(sema.gpa, elements.len);
        const extra_index = @intCast(u32, sema.air_extra.items.len);
        sema.appendRefsAssumeCapacity(elements);

        return block.addInst(.{
            .tag = .aggregate_init,
            .data = .{ .ty_pl = .{
                .ty = ty_ref,
                .payload = extra_index,
            } },
        });
    }

    fn addUnionInit(
        block: *Block,
        union_ty: Type,
        field_index: u32,
        init: Air.Inst.Ref,
    ) !Air.Inst.Ref {
        return block.addInst(.{
            .tag = .union_init,
            .data = .{ .ty_pl = .{
                .ty = try block.sema.addType(union_ty),
                .payload = try block.sema.addExtra(Air.UnionInit{
                    .field_index = field_index,
                    .init = init,
                }),
            } },
        });
    }

    pub fn addInst(block: *Block, inst: Air.Inst) error{OutOfMemory}!Air.Inst.Ref {
        return Air.indexToRef(try block.addInstAsIndex(inst));
    }

    pub fn addInstAsIndex(block: *Block, inst: Air.Inst) error{OutOfMemory}!Air.Inst.Index {
        const sema = block.sema;
        const gpa = sema.gpa;

        try sema.air_instructions.ensureUnusedCapacity(gpa, 1);
        try block.instructions.ensureUnusedCapacity(gpa, 1);

        const result_index = @intCast(Air.Inst.Index, sema.air_instructions.len);
        sema.air_instructions.appendAssumeCapacity(inst);
        block.instructions.appendAssumeCapacity(result_index);
        return result_index;
    }

    fn addUnreachable(block: *Block, src: LazySrcLoc, safety_check: bool) !void {
        if (safety_check and block.wantSafety()) {
            _ = try block.sema.safetyPanic(block, src, .unreach);
        } else {
            _ = try block.addNoOp(.unreach);
        }
    }

    pub fn startAnonDecl(block: *Block, src: LazySrcLoc) !WipAnonDecl {
        return WipAnonDecl{
            .block = block,
            .src = src,
            .new_decl_arena = std.heap.ArenaAllocator.init(block.sema.gpa),
            .finished = false,
        };
    }

    pub const WipAnonDecl = struct {
        block: *Block,
        src: LazySrcLoc,
        new_decl_arena: std.heap.ArenaAllocator,
        finished: bool,

        pub fn arena(wad: *WipAnonDecl) Allocator {
            return wad.new_decl_arena.allocator();
        }

        pub fn deinit(wad: *WipAnonDecl) void {
            if (!wad.finished) {
                wad.new_decl_arena.deinit();
            }
            wad.* = undefined;
        }

        /// `alignment` value of 0 means to use ABI alignment.
        pub fn finish(wad: *WipAnonDecl, ty: Type, val: Value, alignment: u32) !*Decl {
            const sema = wad.block.sema;
            // Do this ahead of time because `createAnonymousDecl` depends on calling
            // `type.hasRuntimeBits()`.
            _ = try sema.typeHasRuntimeBits(wad.block, wad.src, ty);
            const new_decl = try sema.mod.createAnonymousDecl(wad.block, .{
                .ty = ty,
                .val = val,
            });
            new_decl.@"align" = alignment;
            errdefer sema.mod.abortAnonDecl(new_decl);
            try new_decl.finalizeNewArena(&wad.new_decl_arena);
            wad.finished = true;
            return new_decl;
        }
    };
};

pub fn deinit(sema: *Sema) void {
    const gpa = sema.gpa;
    sema.air_instructions.deinit(gpa);
    sema.air_extra.deinit(gpa);
    sema.air_values.deinit(gpa);
    sema.inst_map.deinit(gpa);
    sema.decl_val_table.deinit(gpa);
    sema.types_to_resolve.deinit(gpa);
    sema.* = undefined;
}

/// Returns only the result from the body that is specified.
/// Only appropriate to call when it is determined at comptime that this body
/// has no peers.
fn resolveBody(
    sema: *Sema,
    block: *Block,
    body: []const Zir.Inst.Index,
    /// This is the instruction that a break instruction within `body` can
    /// use to return from the body.
    body_inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const break_data = (try sema.analyzeBodyBreak(block, body)) orelse
        return Air.Inst.Ref.unreachable_value;
    // For comptime control flow, we need to detect when `analyzeBody` reports
    // that we need to break from an outer block. In such case we
    // use Zig's error mechanism to send control flow up the stack until
    // we find the corresponding block to this break.
    if (block.is_comptime and break_data.block_inst != body_inst) {
        sema.comptime_break_inst = break_data.inst;
        return error.ComptimeBreak;
    }
    return sema.resolveInst(break_data.operand);
}

pub fn analyzeBody(
    sema: *Sema,
    block: *Block,
    body: []const Zir.Inst.Index,
) !void {
    _ = sema.analyzeBodyInner(block, body) catch |err| switch (err) {
        error.ComptimeBreak => unreachable, // unexpected comptime control flow
        else => |e| return e,
    };
}

const BreakData = struct {
    block_inst: Zir.Inst.Index,
    operand: Air.Inst.Ref,
    inst: Air.Inst.Index,
};

pub fn analyzeBodyBreak(
    sema: *Sema,
    block: *Block,
    body: []const Zir.Inst.Index,
) CompileError!?BreakData {
    const break_inst = sema.analyzeBodyInner(block, body) catch |err| switch (err) {
        error.ComptimeBreak => sema.comptime_break_inst,
        else => |e| return e,
    };
    if (block.instructions.items.len != 0 and
        sema.typeOf(Air.indexToRef(block.instructions.items[block.instructions.items.len - 1])).isNoReturn())
        return null;
    const break_data = sema.code.instructions.items(.data)[break_inst].@"break";
    return BreakData{
        .block_inst = break_data.block_inst,
        .operand = break_data.operand,
        .inst = break_inst,
    };
}

/// ZIR instructions which are always `noreturn` return this. This matches the
/// return type of `analyzeBody` so that we can tail call them.
/// Only appropriate to return when the instruction is known to be NoReturn
/// solely based on the ZIR tag.
const always_noreturn: CompileError!Zir.Inst.Index = @as(Zir.Inst.Index, undefined);

/// This function is the main loop of `Sema` and it can be used in two different ways:
/// * The traditional way where there are N breaks out of the block and peer type
///   resolution is done on the break operands. In this case, the `Zir.Inst.Index`
///   part of the return value will be `undefined`, and callsites should ignore it,
///   finding the block result value via the block scope.
/// * The "flat" way. There is only 1 break out of the block, and it is with a `break_inline`
///   instruction. In this case, the `Zir.Inst.Index` part of the return value will be
///   the break instruction. This communicates both which block the break applies to, as
///   well as the operand. No block scope needs to be created for this strategy.
fn analyzeBodyInner(
    sema: *Sema,
    block: *Block,
    body: []const Zir.Inst.Index,
) CompileError!Zir.Inst.Index {
    // No tracy calls here, to avoid interfering with the tail call mechanism.

    const parent_capture_scope = block.wip_capture_scope;

    var wip_captures = WipCaptureScope{
        .finalized = true,
        .scope = parent_capture_scope,
        .perm_arena = sema.perm_arena,
        .gpa = sema.gpa,
    };
    defer if (wip_captures.scope != parent_capture_scope) {
        wip_captures.deinit();
    };

    const map = &sema.inst_map;
    const tags = sema.code.instructions.items(.tag);
    const datas = sema.code.instructions.items(.data);

    var orig_captures: usize = parent_capture_scope.captures.count();

    var crash_info = crash_report.prepAnalyzeBody(sema, block, body);
    crash_info.push();
    defer crash_info.pop();

    var dbg_block_begins: u32 = 0;

    // We use a while(true) loop here to avoid a redundant way of breaking out of
    // the loop. The only way to break out of the loop is with a `noreturn`
    // instruction.
    var i: usize = 0;
    const result = while (true) {
        crash_info.setBodyIndex(i);
        const inst = body[i];
        std.log.scoped(.sema_zir).debug("sema ZIR {s} %{d}", .{
            block.src_decl.src_namespace.file_scope.sub_file_path, inst,
        });
        const air_inst: Air.Inst.Ref = switch (tags[inst]) {
            // zig fmt: off
            .alloc                        => try sema.zirAlloc(block, inst),
            .alloc_inferred               => try sema.zirAllocInferred(block, inst, Type.initTag(.inferred_alloc_const)),
            .alloc_inferred_mut           => try sema.zirAllocInferred(block, inst, Type.initTag(.inferred_alloc_mut)),
            .alloc_inferred_comptime      => try sema.zirAllocInferredComptime(inst, Type.initTag(.inferred_alloc_const)),
            .alloc_inferred_comptime_mut  => try sema.zirAllocInferredComptime(inst, Type.initTag(.inferred_alloc_mut)),
            .alloc_mut                    => try sema.zirAllocMut(block, inst),
            .alloc_comptime_mut           => try sema.zirAllocComptime(block, inst),
            .make_ptr_const               => try sema.zirMakePtrConst(block, inst),
            .anyframe_type                => try sema.zirAnyframeType(block, inst),
            .array_cat                    => try sema.zirArrayCat(block, inst),
            .array_mul                    => try sema.zirArrayMul(block, inst),
            .array_type                   => try sema.zirArrayType(block, inst),
            .array_type_sentinel          => try sema.zirArrayTypeSentinel(block, inst),
            .vector_type                  => try sema.zirVectorType(block, inst),
            .as                           => try sema.zirAs(block, inst),
            .as_node                      => try sema.zirAsNode(block, inst),
            .bit_and                      => try sema.zirBitwise(block, inst, .bit_and),
            .bit_not                      => try sema.zirBitNot(block, inst),
            .bit_or                       => try sema.zirBitwise(block, inst, .bit_or),
            .bitcast                      => try sema.zirBitcast(block, inst),
            .suspend_block                => try sema.zirSuspendBlock(block, inst),
            .bool_not                     => try sema.zirBoolNot(block, inst),
            .bool_br_and                  => try sema.zirBoolBr(block, inst, false),
            .bool_br_or                   => try sema.zirBoolBr(block, inst, true),
            .c_import                     => try sema.zirCImport(block, inst),
            .call                         => try sema.zirCall(block, inst),
            .closure_get                  => try sema.zirClosureGet(block, inst),
            .cmp_lt                       => try sema.zirCmp(block, inst, .lt),
            .cmp_lte                      => try sema.zirCmp(block, inst, .lte),
            .cmp_eq                       => try sema.zirCmpEq(block, inst, .eq, .cmp_eq),
            .cmp_gte                      => try sema.zirCmp(block, inst, .gte),
            .cmp_gt                       => try sema.zirCmp(block, inst, .gt),
            .cmp_neq                      => try sema.zirCmpEq(block, inst, .neq, .cmp_neq),
            .coerce_result_ptr            => try sema.zirCoerceResultPtr(block, inst),
            .decl_ref                     => try sema.zirDeclRef(block, inst),
            .decl_val                     => try sema.zirDeclVal(block, inst),
            .load                         => try sema.zirLoad(block, inst),
            .elem_ptr                     => try sema.zirElemPtr(block, inst),
            .elem_ptr_node                => try sema.zirElemPtrNode(block, inst),
            .elem_ptr_imm                 => try sema.zirElemPtrImm(block, inst),
            .elem_val                     => try sema.zirElemVal(block, inst),
            .elem_val_node                => try sema.zirElemValNode(block, inst),
            .elem_type                    => try sema.zirElemType(block, inst),
            .enum_literal                 => try sema.zirEnumLiteral(block, inst),
            .enum_to_int                  => try sema.zirEnumToInt(block, inst),
            .int_to_enum                  => try sema.zirIntToEnum(block, inst),
            .err_union_code               => try sema.zirErrUnionCode(block, inst),
            .err_union_code_ptr           => try sema.zirErrUnionCodePtr(block, inst),
            .err_union_payload_safe       => try sema.zirErrUnionPayload(block, inst, true),
            .err_union_payload_safe_ptr   => try sema.zirErrUnionPayloadPtr(block, inst, true),
            .err_union_payload_unsafe     => try sema.zirErrUnionPayload(block, inst, false),
            .err_union_payload_unsafe_ptr => try sema.zirErrUnionPayloadPtr(block, inst, false),
            .error_union_type             => try sema.zirErrorUnionType(block, inst),
            .error_value                  => try sema.zirErrorValue(block, inst),
            .error_to_int                 => try sema.zirErrorToInt(block, inst),
            .int_to_error                 => try sema.zirIntToError(block, inst),
            .field_ptr                    => try sema.zirFieldPtr(block, inst),
            .field_ptr_named              => try sema.zirFieldPtrNamed(block, inst),
            .field_val                    => try sema.zirFieldVal(block, inst),
            .field_val_named              => try sema.zirFieldValNamed(block, inst),
            .field_call_bind              => try sema.zirFieldCallBind(block, inst),
            .field_call_bind_named        => try sema.zirFieldCallBindNamed(block, inst),
            .func                         => try sema.zirFunc(block, inst, false),
            .func_inferred                => try sema.zirFunc(block, inst, true),
            .import                       => try sema.zirImport(block, inst),
            .indexable_ptr_len            => try sema.zirIndexablePtrLen(block, inst),
            .int                          => try sema.zirInt(block, inst),
            .int_big                      => try sema.zirIntBig(block, inst),
            .float                        => try sema.zirFloat(block, inst),
            .float128                     => try sema.zirFloat128(block, inst),
            .int_type                     => try sema.zirIntType(block, inst),
            .is_non_err                   => try sema.zirIsNonErr(block, inst),
            .is_non_err_ptr               => try sema.zirIsNonErrPtr(block, inst),
            .is_non_null                  => try sema.zirIsNonNull(block, inst),
            .is_non_null_ptr              => try sema.zirIsNonNullPtr(block, inst),
            .merge_error_sets             => try sema.zirMergeErrorSets(block, inst),
            .negate                       => try sema.zirNegate(block, inst, .sub),
            .negate_wrap                  => try sema.zirNegate(block, inst, .subwrap),
            .optional_payload_safe        => try sema.zirOptionalPayload(block, inst, true),
            .optional_payload_safe_ptr    => try sema.zirOptionalPayloadPtr(block, inst, true),
            .optional_payload_unsafe      => try sema.zirOptionalPayload(block, inst, false),
            .optional_payload_unsafe_ptr  => try sema.zirOptionalPayloadPtr(block, inst, false),
            .optional_type                => try sema.zirOptionalType(block, inst),
            .param_type                   => try sema.zirParamType(block, inst),
            .ptr_type                     => try sema.zirPtrType(block, inst),
            .ptr_type_simple              => try sema.zirPtrTypeSimple(block, inst),
            .ref                          => try sema.zirRef(block, inst),
            .ret_err_value_code           => try sema.zirRetErrValueCode(block, inst),
            .shr                          => try sema.zirShr(block, inst, .shr),
            .shr_exact                    => try sema.zirShr(block, inst, .shr_exact),
            .slice_end                    => try sema.zirSliceEnd(block, inst),
            .slice_sentinel               => try sema.zirSliceSentinel(block, inst),
            .slice_start                  => try sema.zirSliceStart(block, inst),
            .str                          => try sema.zirStr(block, inst),
            .switch_block                 => try sema.zirSwitchBlock(block, inst),
            .switch_cond                  => try sema.zirSwitchCond(block, inst, false),
            .switch_cond_ref              => try sema.zirSwitchCond(block, inst, true),
            .switch_capture               => try sema.zirSwitchCapture(block, inst, false, false),
            .switch_capture_ref           => try sema.zirSwitchCapture(block, inst, false, true),
            .switch_capture_multi         => try sema.zirSwitchCapture(block, inst, true, false),
            .switch_capture_multi_ref     => try sema.zirSwitchCapture(block, inst, true, true),
            .type_info                    => try sema.zirTypeInfo(block, inst),
            .size_of                      => try sema.zirSizeOf(block, inst),
            .bit_size_of                  => try sema.zirBitSizeOf(block, inst),
            .typeof                       => try sema.zirTypeof(block, inst),
            .typeof_builtin               => try sema.zirTypeofBuiltin(block, inst),
            .log2_int_type                => try sema.zirLog2IntType(block, inst),
            .typeof_log2_int_type         => try sema.zirTypeofLog2IntType(block, inst),
            .xor                          => try sema.zirBitwise(block, inst, .xor),
            .struct_init_empty            => try sema.zirStructInitEmpty(block, inst),
            .struct_init                  => try sema.zirStructInit(block, inst, false),
            .struct_init_ref              => try sema.zirStructInit(block, inst, true),
            .struct_init_anon             => try sema.zirStructInitAnon(block, inst, false),
            .struct_init_anon_ref         => try sema.zirStructInitAnon(block, inst, true),
            .array_init                   => try sema.zirArrayInit(block, inst, false, false),
            .array_init_sent              => try sema.zirArrayInit(block, inst, false, true),
            .array_init_ref               => try sema.zirArrayInit(block, inst, true, false),
            .array_init_sent_ref          => try sema.zirArrayInit(block, inst, true, true),
            .array_init_anon              => try sema.zirArrayInitAnon(block, inst, false),
            .array_init_anon_ref          => try sema.zirArrayInitAnon(block, inst, true),
            .union_init                   => try sema.zirUnionInit(block, inst),
            .field_type                   => try sema.zirFieldType(block, inst),
            .field_type_ref               => try sema.zirFieldTypeRef(block, inst),
            .ptr_to_int                   => try sema.zirPtrToInt(block, inst),
            .align_of                     => try sema.zirAlignOf(block, inst),
            .bool_to_int                  => try sema.zirBoolToInt(block, inst),
            .embed_file                   => try sema.zirEmbedFile(block, inst),
            .error_name                   => try sema.zirErrorName(block, inst),
            .tag_name                     => try sema.zirTagName(block, inst),
            .reify                        => try sema.zirReify(block, inst),
            .type_name                    => try sema.zirTypeName(block, inst),
            .frame_type                   => try sema.zirFrameType(block, inst),
            .frame_size                   => try sema.zirFrameSize(block, inst),
            .float_to_int                 => try sema.zirFloatToInt(block, inst),
            .int_to_float                 => try sema.zirIntToFloat(block, inst),
            .int_to_ptr                   => try sema.zirIntToPtr(block, inst),
            .float_cast                   => try sema.zirFloatCast(block, inst),
            .int_cast                     => try sema.zirIntCast(block, inst),
            .err_set_cast                 => try sema.zirErrSetCast(block, inst),
            .ptr_cast                     => try sema.zirPtrCast(block, inst),
            .truncate                     => try sema.zirTruncate(block, inst),
            .align_cast                   => try sema.zirAlignCast(block, inst),
            .has_decl                     => try sema.zirHasDecl(block, inst),
            .has_field                    => try sema.zirHasField(block, inst),
            .byte_swap                    => try sema.zirByteSwap(block, inst),
            .bit_reverse                  => try sema.zirBitReverse(block, inst),
            .bit_offset_of                => try sema.zirBitOffsetOf(block, inst),
            .offset_of                    => try sema.zirOffsetOf(block, inst),
            .cmpxchg_strong               => try sema.zirCmpxchg(block, inst, .cmpxchg_strong),
            .cmpxchg_weak                 => try sema.zirCmpxchg(block, inst, .cmpxchg_weak),
            .splat                        => try sema.zirSplat(block, inst),
            .reduce                       => try sema.zirReduce(block, inst),
            .shuffle                      => try sema.zirShuffle(block, inst),
            .select                       => try sema.zirSelect(block, inst),
            .atomic_load                  => try sema.zirAtomicLoad(block, inst),
            .atomic_rmw                   => try sema.zirAtomicRmw(block, inst),
            .mul_add                      => try sema.zirMulAdd(block, inst),
            .builtin_call                 => try sema.zirBuiltinCall(block, inst),
            .field_parent_ptr             => try sema.zirFieldParentPtr(block, inst),
            .builtin_async_call           => try sema.zirBuiltinAsyncCall(block, inst),
            .@"resume"                    => try sema.zirResume(block, inst),
            .@"await"                     => try sema.zirAwait(block, inst, false),
            .await_nosuspend              => try sema.zirAwait(block, inst, true),
            .array_base_ptr               => try sema.zirArrayBasePtr(block, inst),
            .field_base_ptr               => try sema.zirFieldBasePtr(block, inst),

            .clz       => try sema.zirBitCount(block, inst, .clz,      Value.clz),
            .ctz       => try sema.zirBitCount(block, inst, .ctz,      Value.ctz),
            .pop_count => try sema.zirBitCount(block, inst, .popcount, Value.popCount),

            .sqrt  => try sema.zirUnaryMath(block, inst, .sqrt, Value.sqrt),
            .sin   => try sema.zirUnaryMath(block, inst, .sin, Value.sin),
            .cos   => try sema.zirUnaryMath(block, inst, .cos, Value.cos),
            .exp   => try sema.zirUnaryMath(block, inst, .exp, Value.exp),
            .exp2  => try sema.zirUnaryMath(block, inst, .exp2, Value.exp2),
            .log   => try sema.zirUnaryMath(block, inst, .log, Value.log),
            .log2  => try sema.zirUnaryMath(block, inst, .log2, Value.log2),
            .log10 => try sema.zirUnaryMath(block, inst, .log10, Value.log10),
            .fabs  => try sema.zirUnaryMath(block, inst, .fabs, Value.fabs),
            .floor => try sema.zirUnaryMath(block, inst, .floor, Value.floor),
            .ceil  => try sema.zirUnaryMath(block, inst, .ceil, Value.ceil),
            .round => try sema.zirUnaryMath(block, inst, .round, Value.round),
            .trunc => try sema.zirUnaryMath(block, inst, .trunc_float, Value.trunc),

            .error_set_decl      => try sema.zirErrorSetDecl(block, inst, .parent),
            .error_set_decl_anon => try sema.zirErrorSetDecl(block, inst, .anon),
            .error_set_decl_func => try sema.zirErrorSetDecl(block, inst, .func),

            .add       => try sema.zirArithmetic(block, inst, .add),
            .addwrap   => try sema.zirArithmetic(block, inst, .addwrap),
            .add_sat   => try sema.zirArithmetic(block, inst, .add_sat),
            .div       => try sema.zirArithmetic(block, inst, .div),
            .div_exact => try sema.zirArithmetic(block, inst, .div_exact),
            .div_floor => try sema.zirArithmetic(block, inst, .div_floor),
            .div_trunc => try sema.zirArithmetic(block, inst, .div_trunc),
            .mod_rem   => try sema.zirArithmetic(block, inst, .mod_rem),
            .mod       => try sema.zirArithmetic(block, inst, .mod),
            .rem       => try sema.zirArithmetic(block, inst, .rem),
            .mul       => try sema.zirArithmetic(block, inst, .mul),
            .mulwrap   => try sema.zirArithmetic(block, inst, .mulwrap),
            .mul_sat   => try sema.zirArithmetic(block, inst, .mul_sat),
            .sub       => try sema.zirArithmetic(block, inst, .sub),
            .subwrap   => try sema.zirArithmetic(block, inst, .subwrap),
            .sub_sat   => try sema.zirArithmetic(block, inst, .sub_sat),

            .maximum => try sema.zirMinMax(block, inst, .max),
            .minimum => try sema.zirMinMax(block, inst, .min),

            .shl       => try sema.zirShl(block, inst, .shl),
            .shl_exact => try sema.zirShl(block, inst, .shl_exact),
            .shl_sat   => try sema.zirShl(block, inst, .shl_sat),

            // Instructions that we know to *always* be noreturn based solely on their tag.
            // These functions match the return type of analyzeBody so that we can
            // tail call them here.
            .compile_error  => break sema.zirCompileError(block, inst),
            .ret_tok        => break sema.zirRetTok(block, inst),
            .ret_node       => break sema.zirRetNode(block, inst),
            .ret_load       => break sema.zirRetLoad(block, inst),
            .ret_err_value  => break sema.zirRetErrValue(block, inst),
            .@"unreachable" => break sema.zirUnreachable(block, inst),
            .panic          => break sema.zirPanic(block, inst),
            // zig fmt: on

            .extended => ext: {
                const extended = datas[inst].extended;
                break :ext switch (extended.opcode) {
                    // zig fmt: off
                    .func               => try sema.zirFuncExtended(      block, extended, inst),
                    .variable           => try sema.zirVarExtended(       block, extended),
                    .struct_decl        => try sema.zirStructDecl(        block, extended, inst),
                    .enum_decl          => try sema.zirEnumDecl(          block, extended),
                    .union_decl         => try sema.zirUnionDecl(         block, extended, inst),
                    .opaque_decl        => try sema.zirOpaqueDecl(        block, extended),
                    .ret_ptr            => try sema.zirRetPtr(            block, extended),
                    .ret_type           => try sema.zirRetType(           block, extended),
                    .this               => try sema.zirThis(              block, extended),
                    .ret_addr           => try sema.zirRetAddr(           block, extended),
                    .builtin_src        => try sema.zirBuiltinSrc(        block, extended),
                    .error_return_trace => try sema.zirErrorReturnTrace(  block, extended),
                    .frame              => try sema.zirFrame(             block, extended),
                    .frame_address      => try sema.zirFrameAddress(      block, extended),
                    .alloc              => try sema.zirAllocExtended(     block, extended),
                    .builtin_extern     => try sema.zirBuiltinExtern(     block, extended),
                    .@"asm"             => try sema.zirAsm(               block, extended),
                    .typeof_peer        => try sema.zirTypeofPeer(        block, extended),
                    .compile_log        => try sema.zirCompileLog(        block, extended),
                    .add_with_overflow  => try sema.zirOverflowArithmetic(block, extended, extended.opcode),
                    .sub_with_overflow  => try sema.zirOverflowArithmetic(block, extended, extended.opcode),
                    .mul_with_overflow  => try sema.zirOverflowArithmetic(block, extended, extended.opcode),
                    .shl_with_overflow  => try sema.zirOverflowArithmetic(block, extended, extended.opcode),
                    .c_undef            => try sema.zirCUndef(            block, extended),
                    .c_include          => try sema.zirCInclude(          block, extended),
                    .c_define           => try sema.zirCDefine(           block, extended),
                    .wasm_memory_size   => try sema.zirWasmMemorySize(    block, extended),
                    .wasm_memory_grow   => try sema.zirWasmMemoryGrow(    block, extended),
                    .prefetch           => try sema.zirPrefetch(          block, extended),
                    // zig fmt: on
                    .dbg_block_begin => {
                        dbg_block_begins += 1;
                        try sema.zirDbgBlockBegin(block);
                        i += 1;
                        continue;
                    },
                    .dbg_block_end => {
                        dbg_block_begins -= 1;
                        try sema.zirDbgBlockEnd(block);
                        i += 1;
                        continue;
                    },
                };
            },

            // Instructions that we know can *never* be noreturn based solely on
            // their tag. We avoid needlessly checking if they are noreturn and
            // continue the loop.
            // We also know that they cannot be referenced later, so we avoid
            // putting them into the map.
            .breakpoint => {
                if (!block.is_comptime) {
                    _ = try block.addNoOp(.breakpoint);
                }
                i += 1;
                continue;
            },
            .fence => {
                try sema.zirFence(block, inst);
                i += 1;
                continue;
            },
            .dbg_stmt => {
                try sema.zirDbgStmt(block, inst);
                i += 1;
                continue;
            },
            .dbg_var_ptr => {
                try sema.zirDbgVar(block, inst, .dbg_var_ptr);
                i += 1;
                continue;
            },
            .dbg_var_val => {
                try sema.zirDbgVar(block, inst, .dbg_var_val);
                i += 1;
                continue;
            },
            .ensure_err_payload_void => {
                try sema.zirEnsureErrPayloadVoid(block, inst);
                i += 1;
                continue;
            },
            .ensure_result_non_error => {
                try sema.zirEnsureResultNonError(block, inst);
                i += 1;
                continue;
            },
            .ensure_result_used => {
                try sema.zirEnsureResultUsed(block, inst);
                i += 1;
                continue;
            },
            .set_eval_branch_quota => {
                try sema.zirSetEvalBranchQuota(block, inst);
                i += 1;
                continue;
            },
            .atomic_store => {
                try sema.zirAtomicStore(block, inst);
                i += 1;
                continue;
            },
            .store => {
                try sema.zirStore(block, inst);
                i += 1;
                continue;
            },
            .store_node => {
                try sema.zirStoreNode(block, inst);
                i += 1;
                continue;
            },
            .store_to_block_ptr => {
                try sema.zirStoreToBlockPtr(block, inst);
                i += 1;
                continue;
            },
            .store_to_inferred_ptr => {
                try sema.zirStoreToInferredPtr(block, inst);
                i += 1;
                continue;
            },
            .resolve_inferred_alloc => {
                try sema.zirResolveInferredAlloc(block, inst);
                i += 1;
                continue;
            },
            .validate_array_init_ty => {
                try sema.validateArrayInitTy(block, inst);
                i += 1;
                continue;
            },
            .validate_struct_init_ty => {
                try sema.validateStructInitTy(block, inst);
                i += 1;
                continue;
            },
            .validate_struct_init => {
                try sema.zirValidateStructInit(block, inst, false);
                i += 1;
                continue;
            },
            .validate_struct_init_comptime => {
                try sema.zirValidateStructInit(block, inst, true);
                i += 1;
                continue;
            },
            .validate_array_init => {
                try sema.zirValidateArrayInit(block, inst, false);
                i += 1;
                continue;
            },
            .validate_array_init_comptime => {
                try sema.zirValidateArrayInit(block, inst, true);
                i += 1;
                continue;
            },
            .@"export" => {
                try sema.zirExport(block, inst);
                i += 1;
                continue;
            },
            .export_value => {
                try sema.zirExportValue(block, inst);
                i += 1;
                continue;
            },
            .set_align_stack => {
                try sema.zirSetAlignStack(block, inst);
                i += 1;
                continue;
            },
            .set_cold => {
                try sema.zirSetCold(block, inst);
                i += 1;
                continue;
            },
            .set_float_mode => {
                try sema.zirSetFloatMode(block, inst);
                i += 1;
                continue;
            },
            .set_runtime_safety => {
                try sema.zirSetRuntimeSafety(block, inst);
                i += 1;
                continue;
            },
            .param => {
                try sema.zirParam(block, inst, false);
                i += 1;
                continue;
            },
            .param_comptime => {
                try sema.zirParam(block, inst, true);
                i += 1;
                continue;
            },
            .param_anytype => {
                try sema.zirParamAnytype(block, inst, false);
                i += 1;
                continue;
            },
            .param_anytype_comptime => {
                try sema.zirParamAnytype(block, inst, true);
                i += 1;
                continue;
            },
            .closure_capture => {
                try sema.zirClosureCapture(block, inst);
                i += 1;
                continue;
            },
            .memcpy => {
                try sema.zirMemcpy(block, inst);
                i += 1;
                continue;
            },
            .memset => {
                try sema.zirMemset(block, inst);
                i += 1;
                continue;
            },

            // Special case instructions to handle comptime control flow.
            .@"break" => {
                if (block.is_comptime) {
                    break inst; // same as break_inline
                } else {
                    break sema.zirBreak(block, inst);
                }
            },
            .break_inline => {
                if (block.is_comptime) {
                    break inst;
                } else {
                    sema.comptime_break_inst = inst;
                    return error.ComptimeBreak;
                }
            },
            .repeat => {
                if (block.is_comptime) {
                    // Send comptime control flow back to the beginning of this block.
                    const src: LazySrcLoc = .{ .node_offset = datas[inst].node };
                    try sema.emitBackwardBranch(block, src);
                    if (wip_captures.scope.captures.count() != orig_captures) {
                        try wip_captures.reset(parent_capture_scope);
                        block.wip_capture_scope = wip_captures.scope;
                        orig_captures = 0;
                    }
                    i = 0;
                    continue;
                } else {
                    const src_node = sema.code.instructions.items(.data)[inst].node;
                    const src: LazySrcLoc = .{ .node_offset = src_node };
                    try sema.requireRuntimeBlock(block, src);
                    break always_noreturn;
                }
            },
            .repeat_inline => {
                // Send comptime control flow back to the beginning of this block.
                const src: LazySrcLoc = .{ .node_offset = datas[inst].node };
                try sema.emitBackwardBranch(block, src);
                if (wip_captures.scope.captures.count() != orig_captures) {
                    try wip_captures.reset(parent_capture_scope);
                    block.wip_capture_scope = wip_captures.scope;
                    orig_captures = 0;
                }
                i = 0;
                continue;
            },
            .loop => blk: {
                if (!block.is_comptime) break :blk try sema.zirLoop(block, inst);
                // Same as `block_inline`. TODO https://github.com/ziglang/zig/issues/8220
                const inst_data = datas[inst].pl_node;
                const extra = sema.code.extraData(Zir.Inst.Block, inst_data.payload_index);
                const inline_body = sema.code.extra[extra.end..][0..extra.data.body_len];
                const break_data = (try sema.analyzeBodyBreak(block, inline_body)) orelse
                    break always_noreturn;
                if (inst == break_data.block_inst) {
                    break :blk sema.resolveInst(break_data.operand);
                } else {
                    break break_data.inst;
                }
            },
            .block => blk: {
                if (!block.is_comptime) break :blk try sema.zirBlock(block, inst);
                // Same as `block_inline`. TODO https://github.com/ziglang/zig/issues/8220
                const inst_data = datas[inst].pl_node;
                const extra = sema.code.extraData(Zir.Inst.Block, inst_data.payload_index);
                const inline_body = sema.code.extra[extra.end..][0..extra.data.body_len];
                // If this block contains a function prototype, we need to reset the
                // current list of parameters and restore it later.
                // Note: this probably needs to be resolved in a more general manner.
                const prev_params = block.params;
                block.params = .{};
                defer {
                    block.params.deinit(sema.gpa);
                    block.params = prev_params;
                }
                const break_data = (try sema.analyzeBodyBreak(block, inline_body)) orelse
                    break always_noreturn;
                if (inst == break_data.block_inst) {
                    break :blk sema.resolveInst(break_data.operand);
                } else {
                    break break_data.inst;
                }
            },
            .block_inline => blk: {
                // Directly analyze the block body without introducing a new block.
                const inst_data = datas[inst].pl_node;
                const extra = sema.code.extraData(Zir.Inst.Block, inst_data.payload_index);
                const inline_body = sema.code.extra[extra.end..][0..extra.data.body_len];
                // If this block contains a function prototype, we need to reset the
                // current list of parameters and restore it later.
                // Note: this probably needs to be resolved in a more general manner.
                const prev_params = block.params;
                block.params = .{};
                defer {
                    block.params.deinit(sema.gpa);
                    block.params = prev_params;
                }
                const break_data = (try sema.analyzeBodyBreak(block, inline_body)) orelse
                    break always_noreturn;
                if (inst == break_data.block_inst) {
                    break :blk sema.resolveInst(break_data.operand);
                } else {
                    break break_data.inst;
                }
            },
            .condbr => blk: {
                if (!block.is_comptime) break sema.zirCondbr(block, inst);
                // Same as condbr_inline. TODO https://github.com/ziglang/zig/issues/8220
                const inst_data = datas[inst].pl_node;
                const cond_src: LazySrcLoc = .{ .node_offset_if_cond = inst_data.src_node };
                const extra = sema.code.extraData(Zir.Inst.CondBr, inst_data.payload_index);
                const then_body = sema.code.extra[extra.end..][0..extra.data.then_body_len];
                const else_body = sema.code.extra[extra.end + then_body.len ..][0..extra.data.else_body_len];
                const cond = try sema.resolveInstConst(block, cond_src, extra.data.condition);
                const inline_body = if (cond.val.toBool()) then_body else else_body;
                const break_data = (try sema.analyzeBodyBreak(block, inline_body)) orelse
                    break always_noreturn;
                if (inst == break_data.block_inst) {
                    break :blk sema.resolveInst(break_data.operand);
                } else {
                    break break_data.inst;
                }
            },
            .condbr_inline => blk: {
                const inst_data = datas[inst].pl_node;
                const cond_src: LazySrcLoc = .{ .node_offset_if_cond = inst_data.src_node };
                const extra = sema.code.extraData(Zir.Inst.CondBr, inst_data.payload_index);
                const then_body = sema.code.extra[extra.end..][0..extra.data.then_body_len];
                const else_body = sema.code.extra[extra.end + then_body.len ..][0..extra.data.else_body_len];
                const cond = try sema.resolveInstConst(block, cond_src, extra.data.condition);
                const inline_body = if (cond.val.toBool()) then_body else else_body;
                const break_data = (try sema.analyzeBodyBreak(block, inline_body)) orelse
                    break always_noreturn;
                if (inst == break_data.block_inst) {
                    break :blk sema.resolveInst(break_data.operand);
                } else {
                    break break_data.inst;
                }
            },
        };
        if (sema.typeOf(air_inst).isNoReturn())
            break always_noreturn;
        try map.put(sema.gpa, inst, air_inst);
        i += 1;
    } else unreachable;

    // balance out dbg_block_begins in case of early noreturn
    const noreturn_inst = block.instructions.popOrNull();
    while (dbg_block_begins > 0) {
        dbg_block_begins -= 1;
        if (block.is_comptime or sema.mod.comp.bin_file.options.strip) continue;

        _ = try block.addInst(.{
            .tag = .dbg_block_end,
            .data = undefined,
        });
    }
    if (noreturn_inst) |some| try block.instructions.append(sema.gpa, some);

    if (!wip_captures.finalized) {
        try wip_captures.finalize();
        block.wip_capture_scope = parent_capture_scope;
    }

    return result;
}

pub fn resolveInst(sema: *Sema, zir_ref: Zir.Inst.Ref) Air.Inst.Ref {
    var i: usize = @enumToInt(zir_ref);

    // First section of indexes correspond to a set number of constant values.
    if (i < Zir.Inst.Ref.typed_value_map.len) {
        // We intentionally map the same indexes to the same values between ZIR and AIR.
        return zir_ref;
    }
    i -= Zir.Inst.Ref.typed_value_map.len;

    // Finally, the last section of indexes refers to the map of ZIR=>AIR.
    return sema.inst_map.get(@intCast(u32, i)).?;
}

fn resolveConstBool(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) !bool {
    const air_inst = sema.resolveInst(zir_ref);
    const wanted_type = Type.bool;
    const coerced_inst = try sema.coerce(block, wanted_type, air_inst, src);
    const val = try sema.resolveConstValue(block, src, coerced_inst);
    return val.toBool();
}

pub fn resolveConstString(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) ![]u8 {
    const air_inst = sema.resolveInst(zir_ref);
    const wanted_type = Type.initTag(.const_slice_u8);
    const coerced_inst = try sema.coerce(block, wanted_type, air_inst, src);
    const val = try sema.resolveConstValue(block, src, coerced_inst);
    const target = sema.mod.getTarget();
    return val.toAllocatedBytes(wanted_type, sema.arena, target);
}

pub fn resolveType(sema: *Sema, block: *Block, src: LazySrcLoc, zir_ref: Zir.Inst.Ref) !Type {
    const air_inst = sema.resolveInst(zir_ref);
    const ty = try sema.analyzeAsType(block, src, air_inst);
    if (ty.tag() == .generic_poison) return error.GenericPoison;
    return ty;
}

fn analyzeAsType(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    air_inst: Air.Inst.Ref,
) !Type {
    const wanted_type = Type.initTag(.@"type");
    const coerced_inst = try sema.coerce(block, wanted_type, air_inst, src);
    const val = try sema.resolveConstValue(block, src, coerced_inst);
    var buffer: Value.ToTypeBuffer = undefined;
    const ty = val.toType(&buffer);
    return ty.copy(sema.arena);
}

/// May return Value Tags: `variable`, `undef`.
/// See `resolveConstValue` for an alternative.
/// Value Tag `generic_poison` causes `error.GenericPoison` to be returned.
fn resolveValue(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    air_ref: Air.Inst.Ref,
) CompileError!Value {
    if (try sema.resolveMaybeUndefValAllowVariables(block, src, air_ref)) |val| {
        if (val.tag() == .generic_poison) return error.GenericPoison;
        return val;
    }
    return sema.failWithNeededComptime(block, src);
}

/// Value Tag `variable` will cause a compile error.
/// Value Tag `undef` may be returned.
fn resolveConstMaybeUndefVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    inst: Air.Inst.Ref,
) CompileError!Value {
    if (try sema.resolveMaybeUndefValAllowVariables(block, src, inst)) |val| {
        switch (val.tag()) {
            .variable => return sema.failWithNeededComptime(block, src),
            .generic_poison => return error.GenericPoison,
            else => return val,
        }
    }
    return sema.failWithNeededComptime(block, src);
}

/// Will not return Value Tags: `variable`, `undef`. Instead they will emit compile errors.
/// See `resolveValue` for an alternative.
fn resolveConstValue(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    air_ref: Air.Inst.Ref,
) CompileError!Value {
    if (try sema.resolveMaybeUndefValAllowVariables(block, src, air_ref)) |val| {
        switch (val.tag()) {
            .undef => return sema.failWithUseOfUndef(block, src),
            .variable => return sema.failWithNeededComptime(block, src),
            .generic_poison => return error.GenericPoison,
            else => return val,
        }
    }
    return sema.failWithNeededComptime(block, src);
}

/// Value Tag `variable` causes this function to return `null`.
/// Value Tag `undef` causes this function to return a compile error.
fn resolveDefinedValue(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    air_ref: Air.Inst.Ref,
) CompileError!?Value {
    if (try sema.resolveMaybeUndefVal(block, src, air_ref)) |val| {
        if (val.isUndef()) {
            return sema.failWithUseOfUndef(block, src);
        }
        return val;
    }
    return null;
}

/// Value Tag `variable` causes this function to return `null`.
/// Value Tag `undef` causes this function to return the Value.
/// Value Tag `generic_poison` causes `error.GenericPoison` to be returned.
fn resolveMaybeUndefVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    inst: Air.Inst.Ref,
) CompileError!?Value {
    const val = (try sema.resolveMaybeUndefValAllowVariables(block, src, inst)) orelse return null;
    switch (val.tag()) {
        .variable => return null,
        .generic_poison => return error.GenericPoison,
        else => return val,
    }
}

/// Returns all Value tags including `variable` and `undef`.
fn resolveMaybeUndefValAllowVariables(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    inst: Air.Inst.Ref,
) CompileError!?Value {
    // First section of indexes correspond to a set number of constant values.
    var i: usize = @enumToInt(inst);
    if (i < Air.Inst.Ref.typed_value_map.len) {
        return Air.Inst.Ref.typed_value_map[i].val;
    }
    i -= Air.Inst.Ref.typed_value_map.len;

    if (try sema.typeHasOnePossibleValue(block, src, sema.typeOf(inst))) |opv| {
        return opv;
    }
    const air_tags = sema.air_instructions.items(.tag);
    switch (air_tags[i]) {
        .constant => {
            const ty_pl = sema.air_instructions.items(.data)[i].ty_pl;
            return sema.air_values.items[ty_pl.payload];
        },
        .const_ty => {
            return try sema.air_instructions.items(.data)[i].ty.toValue(sema.arena);
        },
        else => return null,
    }
}

fn failWithNeededComptime(sema: *Sema, block: *Block, src: LazySrcLoc) CompileError {
    return sema.fail(block, src, "unable to resolve comptime value", .{});
}

fn failWithUseOfUndef(sema: *Sema, block: *Block, src: LazySrcLoc) CompileError {
    return sema.fail(block, src, "use of undefined value here causes undefined behavior", .{});
}

fn failWithDivideByZero(sema: *Sema, block: *Block, src: LazySrcLoc) CompileError {
    return sema.fail(block, src, "division by zero here causes undefined behavior", .{});
}

fn failWithModRemNegative(sema: *Sema, block: *Block, src: LazySrcLoc, lhs_ty: Type, rhs_ty: Type) CompileError {
    const target = sema.mod.getTarget();
    return sema.fail(block, src, "remainder division with '{}' and '{}': signed integers and floats must use @rem or @mod", .{
        lhs_ty.fmt(target), rhs_ty.fmt(target),
    });
}

fn failWithExpectedOptionalType(sema: *Sema, block: *Block, src: LazySrcLoc, optional_ty: Type) CompileError {
    const target = sema.mod.getTarget();
    return sema.fail(block, src, "expected optional type, found {}", .{optional_ty.fmt(target)});
}

fn failWithArrayInitNotSupported(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError {
    const target = sema.mod.getTarget();
    return sema.fail(block, src, "type '{}' does not support array initialization syntax", .{
        ty.fmt(target),
    });
}

fn failWithStructInitNotSupported(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError {
    const target = sema.mod.getTarget();
    return sema.fail(block, src, "type '{}' does not support struct initialization syntax", .{
        ty.fmt(target),
    });
}

fn failWithErrorSetCodeMissing(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    dest_err_set_ty: Type,
    src_err_set_ty: Type,
) CompileError {
    const target = sema.mod.getTarget();
    return sema.fail(block, src, "expected type '{}', found type '{}'", .{
        dest_err_set_ty.fmt(target), src_err_set_ty.fmt(target),
    });
}

/// We don't return a pointer to the new error note because the pointer
/// becomes invalid when you add another one.
fn errNote(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    parent: *Module.ErrorMsg,
    comptime format: []const u8,
    args: anytype,
) error{OutOfMemory}!void {
    return sema.mod.errNoteNonLazy(src.toSrcLoc(block.src_decl), parent, format, args);
}

fn errMsg(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    comptime format: []const u8,
    args: anytype,
) error{OutOfMemory}!*Module.ErrorMsg {
    return Module.ErrorMsg.create(sema.gpa, src.toSrcLoc(block.src_decl), format, args);
}

pub fn fail(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    comptime format: []const u8,
    args: anytype,
) CompileError {
    const err_msg = try sema.errMsg(block, src, format, args);
    return sema.failWithOwnedErrorMsg(block, err_msg);
}

fn failWithOwnedErrorMsg(sema: *Sema, block: *Block, err_msg: *Module.ErrorMsg) CompileError {
    @setCold(true);

    if (crash_report.is_enabled and sema.mod.comp.debug_compile_errors) {
        std.debug.print("compile error during Sema: {s}, src: {s}:{}\n", .{
            err_msg.msg,
            err_msg.src_loc.file_scope.sub_file_path,
            err_msg.src_loc.lazy,
        });
        crash_report.compilerPanic("unexpected compile error occurred", null);
    }

    const mod = sema.mod;
    if (block.inlining) |some| some.err = err_msg;

    {
        errdefer err_msg.destroy(mod.gpa);
        if (err_msg.src_loc.lazy == .unneeded) {
            return error.NeededSourceLocation;
        }
        try mod.failed_decls.ensureUnusedCapacity(mod.gpa, 1);
        try mod.failed_files.ensureUnusedCapacity(mod.gpa, 1);
    }
    if (sema.owner_func) |func| {
        func.state = .sema_failure;
    } else {
        sema.owner_decl.analysis = .sema_failure;
        sema.owner_decl.generation = mod.generation;
    }
    mod.failed_decls.putAssumeCapacityNoClobber(sema.owner_decl, err_msg);
    return error.AnalysisFail;
}

pub fn resolveAlign(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) !u16 {
    const alignment_big = try sema.resolveInt(block, src, zir_ref, Type.initTag(.u16));
    const alignment = @intCast(u16, alignment_big); // We coerce to u16 in the prev line.
    if (alignment == 0) return sema.fail(block, src, "alignment must be >= 1", .{});
    if (!std.math.isPowerOfTwo(alignment)) {
        return sema.fail(block, src, "alignment value {d} is not a power of two", .{
            alignment,
        });
    }
    return alignment;
}

fn resolveInt(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
    dest_ty: Type,
) !u64 {
    const air_inst = sema.resolveInst(zir_ref);
    const coerced = try sema.coerce(block, dest_ty, air_inst, src);
    const val = try sema.resolveConstValue(block, src, coerced);
    const target = sema.mod.getTarget();
    return (try val.getUnsignedIntAdvanced(target, sema.kit(block, src))).?;
}

// Returns a compile error if the value has tag `variable`. See `resolveInstValue` for
// a function that does not.
pub fn resolveInstConst(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) CompileError!TypedValue {
    const air_ref = sema.resolveInst(zir_ref);
    const val = try sema.resolveConstValue(block, src, air_ref);
    return TypedValue{
        .ty = sema.typeOf(air_ref),
        .val = val,
    };
}

// Value Tag may be `undef` or `variable`.
// See `resolveInstConst` for an alternative.
pub fn resolveInstValue(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) CompileError!TypedValue {
    const air_ref = sema.resolveInst(zir_ref);
    const val = try sema.resolveValue(block, src, air_ref);
    return TypedValue{
        .ty = sema.typeOf(air_ref),
        .val = val,
    };
}

fn zirCoerceResultPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = sema.src;
    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const pointee_ty = try sema.resolveType(block, src, bin_inst.lhs);
    const ptr = sema.resolveInst(bin_inst.rhs);
    const target = sema.mod.getTarget();
    const addr_space = target_util.defaultAddressSpace(target, .local);

    if (Air.refToIndex(ptr)) |ptr_inst| {
        if (sema.air_instructions.items(.tag)[ptr_inst] == .constant) {
            const air_datas = sema.air_instructions.items(.data);
            const ptr_val = sema.air_values.items[air_datas[ptr_inst].ty_pl.payload];
            switch (ptr_val.tag()) {
                .inferred_alloc => {
                    const inferred_alloc = &ptr_val.castTag(.inferred_alloc).?.data;
                    // Add the stored instruction to the set we will use to resolve peer types
                    // for the inferred allocation.
                    // This instruction will not make it to codegen; it is only to participate
                    // in the `stored_inst_list` of the `inferred_alloc`.
                    var trash_block = block.makeSubBlock();
                    defer trash_block.instructions.deinit(sema.gpa);
                    const operand = try trash_block.addBitCast(pointee_ty, .void_value);

                    try inferred_alloc.stored_inst_list.append(sema.arena, operand);

                    try sema.requireRuntimeBlock(block, src);
                    const ptr_ty = try Type.ptr(sema.arena, target, .{
                        .pointee_type = pointee_ty,
                        .@"align" = inferred_alloc.alignment,
                        .@"addrspace" = addr_space,
                    });
                    const bitcasted_ptr = try block.addBitCast(ptr_ty, ptr);
                    return bitcasted_ptr;
                },
                .inferred_alloc_comptime => {
                    const iac = ptr_val.castTag(.inferred_alloc_comptime).?;
                    // There will be only one coerce_result_ptr because we are running at comptime.
                    // The alloc will turn into a Decl.
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    iac.data.decl = try anon_decl.finish(
                        try pointee_ty.copy(anon_decl.arena()),
                        Value.undef,
                        iac.data.alignment,
                    );
                    if (iac.data.alignment != 0) {
                        try sema.resolveTypeLayout(block, src, pointee_ty);
                    }
                    const ptr_ty = try Type.ptr(sema.arena, target, .{
                        .pointee_type = pointee_ty,
                        .@"align" = iac.data.alignment,
                        .@"addrspace" = addr_space,
                    });
                    return sema.addConstant(
                        ptr_ty,
                        try Value.Tag.decl_ref_mut.create(sema.arena, .{
                            .decl = iac.data.decl,
                            .runtime_index = block.runtime_index,
                        }),
                    );
                },
                else => {},
            }
        }
    }

    // Make a dummy store through the pointer to test the coercion.
    // We will then use the generated instructions to decide what
    // kind of transformations to make on the result pointer.
    var trash_block = block.makeSubBlock();
    trash_block.is_comptime = false;
    defer trash_block.instructions.deinit(sema.gpa);

    const dummy_ptr = try trash_block.addTy(.alloc, sema.typeOf(ptr));
    const dummy_operand = try trash_block.addBitCast(pointee_ty, .void_value);
    try sema.storePtr(&trash_block, src, dummy_ptr, dummy_operand);

    {
        const air_tags = sema.air_instructions.items(.tag);

        //std.debug.print("dummy storePtr instructions:\n", .{});
        //for (trash_block.instructions.items) |item| {
        //    std.debug.print("  {s}\n", .{@tagName(air_tags[item])});
        //}

        // The last one is always `store`.
        const trash_inst = trash_block.instructions.items[trash_block.instructions.items.len - 1];
        if (air_tags[trash_inst] != .store) {
            // no store instruction is generated for zero sized types
            assert((try sema.typeHasOnePossibleValue(block, src, pointee_ty)) != null);
        } else {
            trash_block.instructions.items.len -= 1;
            assert(trash_inst == sema.air_instructions.len - 1);
            sema.air_instructions.len -= 1;
        }
    }

    const ptr_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = pointee_ty,
        .@"addrspace" = addr_space,
    });

    var new_ptr = ptr;

    while (true) {
        const air_tags = sema.air_instructions.items(.tag);
        const air_datas = sema.air_instructions.items(.data);
        const trash_inst = trash_block.instructions.pop();
        switch (air_tags[trash_inst]) {
            .bitcast => {
                if (Air.indexToRef(trash_inst) == dummy_operand) {
                    if (try sema.resolveDefinedValue(block, src, new_ptr)) |ptr_val| {
                        return sema.addConstant(ptr_ty, ptr_val);
                    }
                    return sema.bitCast(block, ptr_ty, new_ptr, src);
                }
                const ty_op = air_datas[trash_inst].ty_op;
                const operand_ty = sema.typeOf(ty_op.operand);
                const ptr_operand_ty = try Type.ptr(sema.arena, target, .{
                    .pointee_type = operand_ty,
                    .@"addrspace" = addr_space,
                });
                if (try sema.resolveDefinedValue(block, src, new_ptr)) |ptr_val| {
                    new_ptr = try sema.addConstant(ptr_operand_ty, ptr_val);
                } else {
                    new_ptr = try sema.bitCast(block, ptr_operand_ty, new_ptr, src);
                }
            },
            .wrap_optional => {
                new_ptr = try sema.analyzeOptionalPayloadPtr(block, src, new_ptr, false, true);
            },
            .wrap_errunion_err => {
                return sema.fail(block, src, "TODO coerce_result_ptr wrap_errunion_err", .{});
            },
            .wrap_errunion_payload => {
                new_ptr = try sema.analyzeErrUnionPayloadPtr(block, src, new_ptr, false, true);
            },
            else => {
                if (std.debug.runtime_safety) {
                    std.debug.panic("unexpected AIR tag for coerce_result_ptr: {s}", .{
                        air_tags[trash_inst],
                    });
                } else {
                    unreachable;
                }
            },
        }
    }
}

pub fn analyzeStructDecl(
    sema: *Sema,
    new_decl: *Decl,
    inst: Zir.Inst.Index,
    struct_obj: *Module.Struct,
) SemaError!void {
    const extended = sema.code.instructions.items(.data)[inst].extended;
    assert(extended.opcode == .struct_decl);
    const small = @bitCast(Zir.Inst.StructDecl.Small, extended.small);

    struct_obj.known_non_opv = small.known_non_opv;
    if (small.known_comptime_only) {
        struct_obj.requires_comptime = .yes;
    }

    var extra_index: usize = extended.operand;
    extra_index += @boolToInt(small.has_src_node);
    extra_index += @boolToInt(small.has_body_len);
    extra_index += @boolToInt(small.has_fields_len);
    const decls_len = if (small.has_decls_len) blk: {
        const decls_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk decls_len;
    } else 0;

    _ = try sema.mod.scanNamespace(&struct_obj.namespace, extra_index, decls_len, new_decl);
}

fn zirStructDecl(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const small = @bitCast(Zir.Inst.StructDecl.Small, extended.small);
    const src: LazySrcLoc = if (small.has_src_node) blk: {
        const node_offset = @bitCast(i32, sema.code.extra[extended.operand]);
        break :blk .{ .node_offset = node_offset };
    } else sema.src;

    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const struct_obj = try new_decl_arena_allocator.create(Module.Struct);
    const struct_ty = try Type.Tag.@"struct".create(new_decl_arena_allocator, struct_obj);
    const struct_val = try Value.Tag.ty.create(new_decl_arena_allocator, struct_ty);
    const type_name = try sema.createTypeName(block, small.name_strategy, "struct");
    const new_decl = try sema.mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = struct_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer sema.mod.abortAnonDecl(new_decl);
    struct_obj.* = .{
        .owner_decl = new_decl,
        .fields = .{},
        .node_offset = src.node_offset,
        .zir_index = inst,
        .layout = small.layout,
        .status = .none,
        .known_non_opv = undefined,
        .namespace = .{
            .parent = block.namespace,
            .ty = struct_ty,
            .file_scope = block.getFileScope(),
        },
    };
    std.log.scoped(.module).debug("create struct {*} owned by {*} ({s})", .{
        &struct_obj.namespace, new_decl, new_decl.name,
    });
    try sema.analyzeStructDecl(new_decl, inst, struct_obj);
    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn createTypeName(
    sema: *Sema,
    block: *Block,
    name_strategy: Zir.Inst.NameStrategy,
    anon_prefix: []const u8,
) ![:0]u8 {
    switch (name_strategy) {
        .anon => {
            // It would be neat to have "struct:line:column" but this name has
            // to survive incremental updates, where it may have been shifted down
            // or up to a different line, but unchanged, and thus not unnecessarily
            // semantically analyzed.
            // This name is also used as the key in the parent namespace so it cannot be
            // renamed.
            const name_index = sema.mod.getNextAnonNameIndex();
            return std.fmt.allocPrintZ(sema.gpa, "{s}__{s}_{d}", .{
                block.src_decl.name, anon_prefix, name_index,
            });
        },
        .parent => return sema.gpa.dupeZ(u8, mem.sliceTo(block.src_decl.name, 0)),
        .func => {
            const target = sema.mod.getTarget();
            const fn_info = sema.code.getFnInfo(sema.func.?.zir_body_inst);
            const zir_tags = sema.code.instructions.items(.tag);

            var buf = std.ArrayList(u8).init(sema.gpa);
            defer buf.deinit();
            try buf.appendSlice(mem.sliceTo(block.src_decl.name, 0));
            try buf.appendSlice("(");

            var arg_i: usize = 0;
            for (fn_info.param_body) |zir_inst| switch (zir_tags[zir_inst]) {
                .param, .param_comptime, .param_anytype, .param_anytype_comptime => {
                    const arg = sema.inst_map.get(zir_inst).?;
                    // The comptime call code in analyzeCall already did this, so we're
                    // just repeating it here and it's guaranteed to work.
                    const arg_val = sema.resolveConstMaybeUndefVal(block, .unneeded, arg) catch unreachable;

                    if (arg_i != 0) try buf.appendSlice(",");
                    try buf.writer().print("{}", .{arg_val.fmtValue(sema.typeOf(arg), target)});

                    arg_i += 1;
                    continue;
                },
                else => continue,
            };

            try buf.appendSlice(")");
            return buf.toOwnedSliceSentinel(0);
        },
    }
}

fn zirEnumDecl(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const mod = sema.mod;
    const gpa = sema.gpa;
    const small = @bitCast(Zir.Inst.EnumDecl.Small, extended.small);
    var extra_index: usize = extended.operand;

    const src: LazySrcLoc = if (small.has_src_node) blk: {
        const node_offset = @bitCast(i32, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk .{ .node_offset = node_offset };
    } else sema.src;

    const tag_type_ref = if (small.has_tag_type) blk: {
        const tag_type_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk tag_type_ref;
    } else .none;

    const body_len = if (small.has_body_len) blk: {
        const body_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk body_len;
    } else 0;

    const fields_len = if (small.has_fields_len) blk: {
        const fields_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk fields_len;
    } else 0;

    const decls_len = if (small.has_decls_len) blk: {
        const decls_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk decls_len;
    } else 0;

    var new_decl_arena = std.heap.ArenaAllocator.init(gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const enum_obj = try new_decl_arena_allocator.create(Module.EnumFull);
    const enum_ty_payload = try new_decl_arena_allocator.create(Type.Payload.EnumFull);
    enum_ty_payload.* = .{
        .base = .{ .tag = if (small.nonexhaustive) .enum_nonexhaustive else .enum_full },
        .data = enum_obj,
    };
    const enum_ty = Type.initPayload(&enum_ty_payload.base);
    const enum_val = try Value.Tag.ty.create(new_decl_arena_allocator, enum_ty);
    const type_name = try sema.createTypeName(block, small.name_strategy, "enum");
    const new_decl = try mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = enum_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer mod.abortAnonDecl(new_decl);

    enum_obj.* = .{
        .owner_decl = new_decl,
        .tag_ty = Type.@"null",
        .tag_ty_inferred = true,
        .fields = .{},
        .values = .{},
        .node_offset = src.node_offset,
        .namespace = .{
            .parent = block.namespace,
            .ty = enum_ty,
            .file_scope = block.getFileScope(),
        },
    };
    std.log.scoped(.module).debug("create enum {*} owned by {*} ({s})", .{
        &enum_obj.namespace, new_decl, new_decl.name,
    });

    extra_index = try mod.scanNamespace(&enum_obj.namespace, extra_index, decls_len, new_decl);

    const body = sema.code.extra[extra_index..][0..body_len];
    if (fields_len == 0) {
        assert(body.len == 0);
        if (tag_type_ref != .none) {
            // TODO better source location
            const ty = try sema.resolveType(block, src, tag_type_ref);
            enum_obj.tag_ty = try ty.copy(new_decl_arena_allocator);
            enum_obj.tag_ty_inferred = false;
        }
        try new_decl.finalizeNewArena(&new_decl_arena);
        return sema.analyzeDeclVal(block, src, new_decl);
    }
    extra_index += body.len;

    const bit_bags_count = std.math.divCeil(usize, fields_len, 32) catch unreachable;
    const body_end = extra_index;
    extra_index += bit_bags_count;

    {
        // We create a block for the field type instructions because they
        // may need to reference Decls from inside the enum namespace.
        // Within the field type, default value, and alignment expressions, the "owner decl"
        // should be the enum itself.

        const prev_owner_decl = sema.owner_decl;
        sema.owner_decl = new_decl;
        defer sema.owner_decl = prev_owner_decl;

        const prev_owner_func = sema.owner_func;
        sema.owner_func = null;
        defer sema.owner_func = prev_owner_func;

        const prev_func = sema.func;
        sema.func = null;
        defer sema.func = prev_func;

        var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, new_decl.src_scope);
        defer wip_captures.deinit();

        var enum_block: Block = .{
            .parent = null,
            .sema = sema,
            .src_decl = new_decl,
            .namespace = &enum_obj.namespace,
            .wip_capture_scope = wip_captures.scope,
            .instructions = .{},
            .inlining = null,
            .is_comptime = true,
        };
        defer assert(enum_block.instructions.items.len == 0); // should all be comptime instructions

        if (body.len != 0) {
            try sema.analyzeBody(&enum_block, body);
        }

        try wip_captures.finalize();

        if (tag_type_ref != .none) {
            // TODO better source location
            const ty = try sema.resolveType(block, src, tag_type_ref);
            enum_obj.tag_ty = try ty.copy(new_decl_arena_allocator);
            enum_obj.tag_ty_inferred = false;
        } else {
            const bits = std.math.log2_int_ceil(usize, fields_len);
            enum_obj.tag_ty = try Type.Tag.int_unsigned.create(new_decl_arena_allocator, bits);
            enum_obj.tag_ty_inferred = true;
        }
    }
    const target = mod.getTarget();

    try enum_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);
    const any_values = for (sema.code.extra[body_end..][0..bit_bags_count]) |bag| {
        if (bag != 0) break true;
    } else false;
    if (any_values) {
        try enum_obj.values.ensureTotalCapacityContext(new_decl_arena_allocator, fields_len, .{
            .ty = enum_obj.tag_ty,
            .target = target,
        });
    }

    var bit_bag_index: usize = body_end;
    var cur_bit_bag: u32 = undefined;
    var field_i: u32 = 0;
    var last_tag_val: ?Value = null;
    while (field_i < fields_len) : (field_i += 1) {
        if (field_i % 32 == 0) {
            cur_bit_bag = sema.code.extra[bit_bag_index];
            bit_bag_index += 1;
        }
        const has_tag_value = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;

        const field_name_zir = sema.code.nullTerminatedString(sema.code.extra[extra_index]);
        extra_index += 1;

        // doc comment
        extra_index += 1;

        // This string needs to outlive the ZIR code.
        const field_name = try new_decl_arena_allocator.dupe(u8, field_name_zir);

        const gop = enum_obj.fields.getOrPutAssumeCapacity(field_name);
        if (gop.found_existing) {
            const tree = try sema.getAstTree(block);
            const field_src = enumFieldSrcLoc(block.src_decl, tree.*, src.node_offset, field_i);
            const other_tag_src = enumFieldSrcLoc(block.src_decl, tree.*, src.node_offset, gop.index);
            const msg = msg: {
                const msg = try sema.errMsg(block, field_src, "duplicate enum tag", .{});
                errdefer msg.destroy(gpa);
                try sema.errNote(block, other_tag_src, msg, "other tag here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }

        if (has_tag_value) {
            const tag_val_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
            extra_index += 1;
            // TODO: if we need to report an error here, use a source location
            // that points to this default value expression rather than the struct.
            // But only resolve the source location if we need to emit a compile error.
            const tag_val = (try sema.resolveInstConst(block, src, tag_val_ref)).val;
            last_tag_val = tag_val;
            const copied_tag_val = try tag_val.copy(new_decl_arena_allocator);
            enum_obj.values.putAssumeCapacityNoClobberContext(copied_tag_val, {}, .{
                .ty = enum_obj.tag_ty,
                .target = target,
            });
        } else if (any_values) {
            const tag_val = if (last_tag_val) |val|
                try val.intAdd(Value.one, enum_obj.tag_ty, sema.arena, target)
            else
                Value.zero;
            last_tag_val = tag_val;
            const copied_tag_val = try tag_val.copy(new_decl_arena_allocator);
            enum_obj.values.putAssumeCapacityNoClobberContext(copied_tag_val, {}, .{
                .ty = enum_obj.tag_ty,
                .target = target,
            });
        }
    }

    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn zirUnionDecl(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const small = @bitCast(Zir.Inst.UnionDecl.Small, extended.small);
    var extra_index: usize = extended.operand;

    const src: LazySrcLoc = if (small.has_src_node) blk: {
        const node_offset = @bitCast(i32, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk .{ .node_offset = node_offset };
    } else sema.src;

    extra_index += @boolToInt(small.has_tag_type);
    extra_index += @boolToInt(small.has_body_len);
    extra_index += @boolToInt(small.has_fields_len);

    const decls_len = if (small.has_decls_len) blk: {
        const decls_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk decls_len;
    } else 0;

    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const union_obj = try new_decl_arena_allocator.create(Module.Union);
    const type_tag: Type.Tag = if (small.has_tag_type or small.auto_enum_tag) .union_tagged else .@"union";
    const union_payload = try new_decl_arena_allocator.create(Type.Payload.Union);
    union_payload.* = .{
        .base = .{ .tag = type_tag },
        .data = union_obj,
    };
    const union_ty = Type.initPayload(&union_payload.base);
    const union_val = try Value.Tag.ty.create(new_decl_arena_allocator, union_ty);
    const type_name = try sema.createTypeName(block, small.name_strategy, "union");
    const new_decl = try sema.mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = union_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer sema.mod.abortAnonDecl(new_decl);
    union_obj.* = .{
        .owner_decl = new_decl,
        .tag_ty = Type.initTag(.@"null"),
        .fields = .{},
        .node_offset = src.node_offset,
        .zir_index = inst,
        .layout = small.layout,
        .status = .none,
        .namespace = .{
            .parent = block.namespace,
            .ty = union_ty,
            .file_scope = block.getFileScope(),
        },
    };
    std.log.scoped(.module).debug("create union {*} owned by {*} ({s})", .{
        &union_obj.namespace, new_decl, new_decl.name,
    });

    _ = try sema.mod.scanNamespace(&union_obj.namespace, extra_index, decls_len, new_decl);

    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn zirOpaqueDecl(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const mod = sema.mod;
    const gpa = sema.gpa;
    const small = @bitCast(Zir.Inst.OpaqueDecl.Small, extended.small);
    var extra_index: usize = extended.operand;

    const src: LazySrcLoc = if (small.has_src_node) blk: {
        const node_offset = @bitCast(i32, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk .{ .node_offset = node_offset };
    } else sema.src;

    const decls_len = if (small.has_decls_len) blk: {
        const decls_len = sema.code.extra[extra_index];
        extra_index += 1;
        break :blk decls_len;
    } else 0;

    var new_decl_arena = std.heap.ArenaAllocator.init(gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const opaque_obj = try new_decl_arena_allocator.create(Module.Opaque);
    const opaque_ty_payload = try new_decl_arena_allocator.create(Type.Payload.Opaque);
    opaque_ty_payload.* = .{
        .base = .{ .tag = .@"opaque" },
        .data = opaque_obj,
    };
    const opaque_ty = Type.initPayload(&opaque_ty_payload.base);
    const opaque_val = try Value.Tag.ty.create(new_decl_arena_allocator, opaque_ty);
    const type_name = try sema.createTypeName(block, small.name_strategy, "opaque");
    const new_decl = try mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = opaque_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer mod.abortAnonDecl(new_decl);

    opaque_obj.* = .{
        .owner_decl = new_decl,
        .node_offset = src.node_offset,
        .namespace = .{
            .parent = block.namespace,
            .ty = opaque_ty,
            .file_scope = block.getFileScope(),
        },
    };
    std.log.scoped(.module).debug("create opaque {*} owned by {*} ({s})", .{
        &opaque_obj.namespace, new_decl, new_decl.name,
    });

    extra_index = try mod.scanNamespace(&opaque_obj.namespace, extra_index, decls_len, new_decl);

    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn zirErrorSetDecl(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    name_strategy: Zir.Inst.NameStrategy,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const gpa = sema.gpa;
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.ErrorSetDecl, inst_data.payload_index);

    var new_decl_arena = std.heap.ArenaAllocator.init(gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const error_set = try new_decl_arena_allocator.create(Module.ErrorSet);
    const error_set_ty = try Type.Tag.error_set.create(new_decl_arena_allocator, error_set);
    const error_set_val = try Value.Tag.ty.create(new_decl_arena_allocator, error_set_ty);
    const type_name = try sema.createTypeName(block, name_strategy, "error");
    const new_decl = try sema.mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = error_set_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer sema.mod.abortAnonDecl(new_decl);

    var names = Module.ErrorSet.NameMap{};
    try names.ensureUnusedCapacity(new_decl_arena_allocator, extra.data.fields_len);

    var extra_index = @intCast(u32, extra.end);
    const extra_index_end = extra_index + (extra.data.fields_len * 2);
    while (extra_index < extra_index_end) : (extra_index += 2) { // +2 to skip over doc_string
        const str_index = sema.code.extra[extra_index];
        const kv = try sema.mod.getErrorValue(sema.code.nullTerminatedString(str_index));
        const result = names.getOrPutAssumeCapacity(kv.key);
        assert(!result.found_existing); // verified in AstGen
    }

    // names must be sorted.
    Module.ErrorSet.sortNames(&names);

    error_set.* = .{
        .owner_decl = new_decl,
        .node_offset = inst_data.src_node,
        .names = names,
    };
    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn zirRetPtr(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    try sema.requireFunctionBlock(block, src);

    if (block.is_comptime) {
        const fn_ret_ty = try sema.resolveTypeFields(block, src, sema.fn_ret_ty);
        return sema.analyzeComptimeAlloc(block, fn_ret_ty, 0, src);
    }

    const target = sema.mod.getTarget();
    const ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = sema.fn_ret_ty,
        .@"addrspace" = target_util.defaultAddressSpace(target, .local),
    });

    if (block.inlining != null) {
        // We are inlining a function call; this should be emitted as an alloc, not a ret_ptr.
        // TODO when functions gain result location support, the inlining struct in
        // Block should contain the return pointer, and we would pass that through here.
        return block.addTy(.alloc, ptr_type);
    }

    return block.addTy(.ret_ptr, ptr_type);
}

fn zirRef(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_tok;
    const operand = sema.resolveInst(inst_data.operand);
    return sema.analyzeRef(block, inst_data.src(), operand);
}

fn zirRetType(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    try sema.requireFunctionBlock(block, src);
    return sema.addType(sema.fn_ret_ty);
}

fn zirEnsureResultUsed(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();

    return sema.ensureResultUsed(block, operand, src);
}

fn ensureResultUsed(
    sema: *Sema,
    block: *Block,
    operand: Air.Inst.Ref,
    src: LazySrcLoc,
) CompileError!void {
    const operand_ty = sema.typeOf(operand);
    switch (operand_ty.zigTypeTag()) {
        .Void, .NoReturn => return,
        else => return sema.fail(block, src, "expression value is ignored", .{}),
    }
}

fn zirEnsureResultNonError(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();
    const operand_ty = sema.typeOf(operand);
    switch (operand_ty.zigTypeTag()) {
        .ErrorSet, .ErrorUnion => return sema.fail(block, src, "error is discarded", .{}),
        else => return,
    }
}

fn zirIndexablePtrLen(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const object = sema.resolveInst(inst_data.operand);
    const object_ty = sema.typeOf(object);

    const is_pointer_to = object_ty.isSinglePointer();

    const array_ty = if (is_pointer_to)
        object_ty.childType()
    else
        object_ty;

    const target = sema.mod.getTarget();
    if (!array_ty.isIndexable()) {
        const msg = msg: {
            const msg = try sema.errMsg(
                block,
                src,
                "type '{}' does not support indexing",
                .{array_ty.fmt(target)},
            );
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(
                block,
                src,
                msg,
                "for loop operand must be an array, slice, tuple, or vector",
                .{},
            );
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    return sema.fieldVal(block, src, object, "len", src);
}

fn zirAllocExtended(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.AllocExtended, extended.operand);
    const src: LazySrcLoc = .{ .node_offset = extra.data.src_node };
    const ty_src = src; // TODO better source location
    const align_src = src; // TODO better source location
    const small = @bitCast(Zir.Inst.AllocExtended.Small, extended.small);

    var extra_index: usize = extra.end;

    const var_ty: Type = if (small.has_type) blk: {
        const type_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk try sema.resolveType(block, ty_src, type_ref);
    } else undefined;

    const alignment: u16 = if (small.has_align) blk: {
        const align_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        const alignment = try sema.resolveAlign(block, align_src, align_ref);
        break :blk alignment;
    } else 0;

    const inferred_alloc_ty = if (small.is_const)
        Type.initTag(.inferred_alloc_const)
    else
        Type.initTag(.inferred_alloc_mut);

    if (block.is_comptime or small.is_comptime) {
        if (small.has_type) {
            return sema.analyzeComptimeAlloc(block, var_ty, alignment, ty_src);
        } else {
            return sema.addConstant(
                inferred_alloc_ty,
                try Value.Tag.inferred_alloc_comptime.create(sema.arena, .{
                    .decl = undefined,
                    .alignment = alignment,
                }),
            );
        }
    }

    if (small.has_type) {
        if (!small.is_const) {
            try sema.validateVarType(block, ty_src, var_ty, false);
        }
        const target = sema.mod.getTarget();
        try sema.requireRuntimeBlock(block, src);
        try sema.resolveTypeLayout(block, src, var_ty);
        const ptr_type = try Type.ptr(sema.arena, target, .{
            .pointee_type = var_ty,
            .@"align" = alignment,
            .@"addrspace" = target_util.defaultAddressSpace(target, .local),
        });
        return block.addTy(.alloc, ptr_type);
    }

    // `Sema.addConstant` does not add the instruction to the block because it is
    // not needed in the case of constant values. However here, we plan to "downgrade"
    // to a normal instruction when we hit `resolve_inferred_alloc`. So we append
    // to the block even though it is currently a `.constant`.
    const result = try sema.addConstant(
        inferred_alloc_ty,
        try Value.Tag.inferred_alloc.create(sema.arena, .{ .alignment = alignment }),
    );
    try sema.requireFunctionBlock(block, src);
    try block.instructions.append(sema.gpa, Air.refToIndex(result).?);
    return result;
}

fn zirAllocComptime(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ty_src: LazySrcLoc = .{ .node_offset_var_decl_ty = inst_data.src_node };
    const var_ty = try sema.resolveType(block, ty_src, inst_data.operand);
    return sema.analyzeComptimeAlloc(block, var_ty, 0, ty_src);
}

fn zirMakePtrConst(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ptr = sema.resolveInst(inst_data.operand);
    const ptr_ty = sema.typeOf(ptr);
    var ptr_info = ptr_ty.ptrInfo().data;
    ptr_info.mutable = false;
    const const_ptr_ty = try Type.ptr(sema.arena, sema.mod.getTarget(), ptr_info);

    if (try sema.resolveMaybeUndefVal(block, inst_data.src(), ptr)) |val| {
        return sema.addConstant(const_ptr_ty, val);
    }
    try sema.requireRuntimeBlock(block, inst_data.src());
    return block.addBitCast(const_ptr_ty, ptr);
}

fn zirAllocInferredComptime(
    sema: *Sema,
    inst: Zir.Inst.Index,
    inferred_alloc_ty: Type,
) CompileError!Air.Inst.Ref {
    const src_node = sema.code.instructions.items(.data)[inst].node;
    const src: LazySrcLoc = .{ .node_offset = src_node };
    sema.src = src;
    return sema.addConstant(
        inferred_alloc_ty,
        try Value.Tag.inferred_alloc_comptime.create(sema.arena, .{
            .decl = undefined,
            .alignment = 0,
        }),
    );
}

fn zirAlloc(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ty_src: LazySrcLoc = .{ .node_offset_var_decl_ty = inst_data.src_node };
    const var_decl_src = inst_data.src();
    const var_ty = try sema.resolveType(block, ty_src, inst_data.operand);
    if (block.is_comptime) {
        return sema.analyzeComptimeAlloc(block, var_ty, 0, ty_src);
    }
    const target = sema.mod.getTarget();
    const ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = var_ty,
        .@"addrspace" = target_util.defaultAddressSpace(target, .local),
    });
    try sema.requireRuntimeBlock(block, var_decl_src);
    try sema.queueFullTypeResolution(var_ty);
    return block.addTy(.alloc, ptr_type);
}

fn zirAllocMut(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const var_decl_src = inst_data.src();
    const ty_src: LazySrcLoc = .{ .node_offset_var_decl_ty = inst_data.src_node };
    const var_ty = try sema.resolveType(block, ty_src, inst_data.operand);
    if (block.is_comptime) {
        return sema.analyzeComptimeAlloc(block, var_ty, 0, ty_src);
    }
    try sema.validateVarType(block, ty_src, var_ty, false);
    const target = sema.mod.getTarget();
    const ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = var_ty,
        .@"addrspace" = target_util.defaultAddressSpace(target, .local),
    });
    try sema.requireRuntimeBlock(block, var_decl_src);
    try sema.queueFullTypeResolution(var_ty);
    return block.addTy(.alloc, ptr_type);
}

fn zirAllocInferred(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    inferred_alloc_ty: Type,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src_node = sema.code.instructions.items(.data)[inst].node;
    const src: LazySrcLoc = .{ .node_offset = src_node };
    sema.src = src;

    if (block.is_comptime) {
        return sema.addConstant(
            inferred_alloc_ty,
            try Value.Tag.inferred_alloc_comptime.create(sema.arena, .{
                .decl = undefined,
                .alignment = 0,
            }),
        );
    }

    // `Sema.addConstant` does not add the instruction to the block because it is
    // not needed in the case of constant values. However here, we plan to "downgrade"
    // to a normal instruction when we hit `resolve_inferred_alloc`. So we append
    // to the block even though it is currently a `.constant`.
    const result = try sema.addConstant(
        inferred_alloc_ty,
        try Value.Tag.inferred_alloc.create(sema.arena, .{ .alignment = 0 }),
    );
    try sema.requireFunctionBlock(block, src);
    try block.instructions.append(sema.gpa, Air.refToIndex(result).?);
    return result;
}

fn zirResolveInferredAlloc(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ty_src: LazySrcLoc = .{ .node_offset_var_decl_ty = inst_data.src_node };
    const ptr = sema.resolveInst(inst_data.operand);
    const ptr_inst = Air.refToIndex(ptr).?;
    assert(sema.air_instructions.items(.tag)[ptr_inst] == .constant);
    const value_index = sema.air_instructions.items(.data)[ptr_inst].ty_pl.payload;
    const ptr_val = sema.air_values.items[value_index];
    const var_is_mut = switch (sema.typeOf(ptr).tag()) {
        .inferred_alloc_const => false,
        .inferred_alloc_mut => true,
        else => unreachable,
    };
    const target = sema.mod.getTarget();

    switch (ptr_val.tag()) {
        .inferred_alloc_comptime => {
            const iac = ptr_val.castTag(.inferred_alloc_comptime).?;
            const decl = iac.data.decl;
            try sema.mod.declareDeclDependency(sema.owner_decl, decl);

            const final_elem_ty = try decl.ty.copy(sema.arena);
            const final_ptr_ty = try Type.ptr(sema.arena, target, .{
                .pointee_type = final_elem_ty,
                .mutable = var_is_mut,
                .@"align" = iac.data.alignment,
                .@"addrspace" = target_util.defaultAddressSpace(target, .local),
            });
            const final_ptr_ty_inst = try sema.addType(final_ptr_ty);
            sema.air_instructions.items(.data)[ptr_inst].ty_pl.ty = final_ptr_ty_inst;

            if (var_is_mut) {
                sema.air_values.items[value_index] = try Value.Tag.decl_ref_mut.create(sema.arena, .{
                    .decl = decl,
                    .runtime_index = block.runtime_index,
                });
            } else {
                sema.air_values.items[value_index] = try Value.Tag.decl_ref.create(sema.arena, decl);
            }
        },
        .inferred_alloc => {
            const inferred_alloc = ptr_val.castTag(.inferred_alloc).?;
            const peer_inst_list = inferred_alloc.data.stored_inst_list.items;
            const final_elem_ty = try sema.resolvePeerTypes(block, ty_src, peer_inst_list, .none);

            const final_ptr_ty = try Type.ptr(sema.arena, target, .{
                .pointee_type = final_elem_ty,
                .mutable = var_is_mut,
                .@"align" = inferred_alloc.data.alignment,
                .@"addrspace" = target_util.defaultAddressSpace(target, .local),
            });

            if (var_is_mut) {
                try sema.validateVarType(block, ty_src, final_elem_ty, false);
            } else ct: {
                // Detect if the value is comptime known. In such case, the
                // last 3 AIR instructions of the block will look like this:
                //
                //   %a = constant
                //   %b = bitcast(%a)
                //   %c = store(%b, %d)
                //
                // If `%d` is comptime-known, then we want to store the value
                // inside an anonymous Decl and then erase these three AIR
                // instructions from the block, replacing the inst_map entry
                // corresponding to the ZIR alloc instruction with a constant
                // decl_ref pointing at our new Decl.
                // dbg_stmt instructions may be interspersed into this pattern
                // which must be ignored.
                if (block.instructions.items.len < 3) break :ct;
                var search_index: usize = block.instructions.items.len;
                const air_tags = sema.air_instructions.items(.tag);
                const air_datas = sema.air_instructions.items(.data);

                const store_inst = while (true) {
                    if (search_index == 0) break :ct;
                    search_index -= 1;

                    const candidate = block.instructions.items[search_index];
                    switch (air_tags[candidate]) {
                        .dbg_stmt => continue,
                        .store => break candidate,
                        else => break :ct,
                    }
                } else unreachable; // TODO shouldn't need this

                const bitcast_inst = while (true) {
                    if (search_index == 0) break :ct;
                    search_index -= 1;

                    const candidate = block.instructions.items[search_index];
                    switch (air_tags[candidate]) {
                        .dbg_stmt => continue,
                        .bitcast => break candidate,
                        else => break :ct,
                    }
                } else unreachable; // TODO shouldn't need this

                const const_inst = while (true) {
                    if (search_index == 0) break :ct;
                    search_index -= 1;

                    const candidate = block.instructions.items[search_index];
                    switch (air_tags[candidate]) {
                        .dbg_stmt => continue,
                        .constant => break candidate,
                        else => break :ct,
                    }
                } else unreachable; // TODO shouldn't need this

                const store_op = air_datas[store_inst].bin_op;
                const store_val = (try sema.resolveMaybeUndefVal(block, src, store_op.rhs)) orelse break :ct;
                if (store_op.lhs != Air.indexToRef(bitcast_inst)) break :ct;
                if (air_datas[bitcast_inst].ty_op.operand != Air.indexToRef(const_inst)) break :ct;

                const new_decl = d: {
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    const new_decl = try anon_decl.finish(
                        try final_elem_ty.copy(anon_decl.arena()),
                        try store_val.copy(anon_decl.arena()),
                        inferred_alloc.data.alignment,
                    );
                    break :d new_decl;
                };
                try sema.mod.declareDeclDependency(sema.owner_decl, new_decl);

                // Even though we reuse the constant instruction, we still remove it from the
                // block so that codegen does not see it.
                block.instructions.shrinkRetainingCapacity(block.instructions.items.len - 3);
                sema.air_values.items[value_index] = try Value.Tag.decl_ref.create(sema.arena, new_decl);
                // if bitcast ty ref needs to be made const, make_ptr_const
                // ZIR handles it later, so we can just use the ty ref here.
                air_datas[ptr_inst].ty_pl.ty = air_datas[bitcast_inst].ty_op.ty;

                // Unless the block is comptime, `alloc_inferred` always produces
                // a runtime constant. The final inferred type needs to be
                // fully resolved so it can be lowered in codegen.
                try sema.resolveTypeFully(block, ty_src, final_elem_ty);

                return;
            }

            try sema.requireRuntimeBlock(block, src);
            try sema.queueFullTypeResolution(final_elem_ty);

            // Change it to a normal alloc.
            sema.air_instructions.set(ptr_inst, .{
                .tag = .alloc,
                .data = .{ .ty = final_ptr_ty },
            });
        },
        else => unreachable,
    }
}

fn zirArrayBasePtr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();

    const start_ptr = sema.resolveInst(inst_data.operand);
    var base_ptr = start_ptr;
    while (true) switch (sema.typeOf(base_ptr).childType().zigTypeTag()) {
        .ErrorUnion => base_ptr = try sema.analyzeErrUnionPayloadPtr(block, src, base_ptr, false, true),
        .Optional => base_ptr = try sema.analyzeOptionalPayloadPtr(block, src, base_ptr, false, true),
        else => break,
    };

    const elem_ty = sema.typeOf(base_ptr).childType();
    switch (elem_ty.zigTypeTag()) {
        .Array, .Vector => return base_ptr,
        .Struct => if (elem_ty.isTuple()) return base_ptr,
        else => {},
    }
    return sema.failWithArrayInitNotSupported(block, src, sema.typeOf(start_ptr).childType());
}

fn zirFieldBasePtr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();

    const start_ptr = sema.resolveInst(inst_data.operand);
    var base_ptr = start_ptr;
    while (true) switch (sema.typeOf(base_ptr).childType().zigTypeTag()) {
        .ErrorUnion => base_ptr = try sema.analyzeErrUnionPayloadPtr(block, src, base_ptr, false, true),
        .Optional => base_ptr = try sema.analyzeOptionalPayloadPtr(block, src, base_ptr, false, true),
        else => break,
    };

    const elem_ty = sema.typeOf(base_ptr).childType();
    switch (elem_ty.zigTypeTag()) {
        .Struct, .Union => return base_ptr,
        else => {},
    }
    return sema.failWithStructInitNotSupported(block, src, sema.typeOf(start_ptr).childType());
}

fn validateArrayInitTy(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ty = try sema.resolveType(block, src, inst_data.operand);

    switch (ty.zigTypeTag()) {
        .Array, .Vector => return,
        .Struct => if (ty.isTuple()) return,
        else => {},
    }
    return sema.failWithArrayInitNotSupported(block, src, ty);
}

fn validateStructInitTy(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ty = try sema.resolveType(block, src, inst_data.operand);

    switch (ty.zigTypeTag()) {
        .Struct, .Union => return,
        else => {},
    }
    return sema.failWithStructInitNotSupported(block, src, ty);
}

fn zirValidateStructInit(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_comptime: bool,
) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const validate_inst = sema.code.instructions.items(.data)[inst].pl_node;
    const init_src = validate_inst.src();
    const validate_extra = sema.code.extraData(Zir.Inst.Block, validate_inst.payload_index);
    const instrs = sema.code.extra[validate_extra.end..][0..validate_extra.data.body_len];
    const field_ptr_data = sema.code.instructions.items(.data)[instrs[0]].pl_node;
    const field_ptr_extra = sema.code.extraData(Zir.Inst.Field, field_ptr_data.payload_index).data;
    const object_ptr = sema.resolveInst(field_ptr_extra.lhs);
    const agg_ty = sema.typeOf(object_ptr).childType();
    switch (agg_ty.zigTypeTag()) {
        .Struct => return sema.validateStructInit(
            block,
            agg_ty.castTag(.@"struct").?.data,
            init_src,
            instrs,
            is_comptime,
        ),
        .Union => return sema.validateUnionInit(
            block,
            agg_ty,
            init_src,
            instrs,
            object_ptr,
            is_comptime,
        ),
        else => unreachable,
    }
}

fn validateUnionInit(
    sema: *Sema,
    block: *Block,
    union_ty: Type,
    init_src: LazySrcLoc,
    instrs: []const Zir.Inst.Index,
    union_ptr: Air.Inst.Ref,
    is_comptime: bool,
) CompileError!void {
    const union_obj = union_ty.cast(Type.Payload.Union).?.data;

    if (instrs.len != 1) {
        const msg = msg: {
            const msg = try sema.errMsg(
                block,
                init_src,
                "cannot initialize multiple union fields at once, unions can only have one active field",
                .{},
            );
            errdefer msg.destroy(sema.gpa);

            for (instrs[1..]) |inst| {
                const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
                const inst_src: LazySrcLoc = .{ .node_offset_back2tok = inst_data.src_node };
                try sema.errNote(block, inst_src, msg, "additional initializer here", .{});
            }
            try sema.addDeclaredHereNote(msg, union_ty);
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    if (is_comptime or block.is_comptime) {
        // In this case, comptime machinery already did everything. No work to do here.
        return;
    }

    const field_ptr = instrs[0];
    const field_ptr_data = sema.code.instructions.items(.data)[field_ptr].pl_node;
    const field_src: LazySrcLoc = .{ .node_offset_back2tok = field_ptr_data.src_node };
    const field_ptr_extra = sema.code.extraData(Zir.Inst.Field, field_ptr_data.payload_index).data;
    const field_name = sema.code.nullTerminatedString(field_ptr_extra.field_name_start);
    const field_index = try sema.unionFieldIndex(block, union_ty, field_name, field_src);
    const air_tags = sema.air_instructions.items(.tag);
    const air_datas = sema.air_instructions.items(.data);
    const field_ptr_air_ref = sema.inst_map.get(field_ptr).?;
    const field_ptr_air_inst = Air.refToIndex(field_ptr_air_ref).?;

    // Our task here is to determine if the union is comptime-known. In such case,
    // we erase the runtime AIR instructions for initializing the union, and replace
    // the mapping with the comptime value. Either way, we will need to populate the tag.

    // We expect to see something like this in the current block AIR:
    //   %a = alloc(*const U)
    //   %b = bitcast(*U, %a)
    //   %c = field_ptr(..., %b)
    //   %e!= store(%c!, %d!)
    // If %d is a comptime operand, the union is comptime.
    // If the union is comptime, we want `first_block_index`
    // to point at %c so that the bitcast becomes the last instruction in the block.
    //
    // In the case of a comptime-known pointer to a union, the
    // the field_ptr instruction is missing, so we have to pattern-match
    // based only on the store instructions.
    // `first_block_index` needs to point to the `field_ptr` if it exists;
    // the `store` otherwise.
    //
    // It's also possible for there to be no store instruction, in the case
    // of nested `coerce_result_ptr` instructions. If we see the `field_ptr`
    // but we have not found a `store`, treat as a runtime-known field.
    var first_block_index = block.instructions.items.len;
    var block_index = block.instructions.items.len - 1;
    var init_val: ?Value = null;
    while (block_index > 0) : (block_index -= 1) {
        const store_inst = block.instructions.items[block_index];
        if (store_inst == field_ptr_air_inst) break;
        if (air_tags[store_inst] != .store) continue;
        const bin_op = air_datas[store_inst].bin_op;
        if (bin_op.lhs != field_ptr_air_ref) continue;
        if (block_index > 0 and
            field_ptr_air_inst == block.instructions.items[block_index - 1])
        {
            first_block_index = @minimum(first_block_index, block_index - 1);
        } else {
            first_block_index = @minimum(first_block_index, block_index);
        }
        init_val = try sema.resolveMaybeUndefValAllowVariables(block, init_src, bin_op.rhs);
        break;
    }

    const tag_val = try Value.Tag.enum_field_index.create(sema.arena, field_index);

    if (init_val) |val| {
        // Our task is to delete all the `field_ptr` and `store` instructions, and insert
        // instead a single `store` to the result ptr with a comptime union value.
        block.instructions.shrinkRetainingCapacity(first_block_index);

        const union_val = try Value.Tag.@"union".create(sema.arena, .{
            .tag = tag_val,
            .val = val,
        });
        const union_init = try sema.addConstant(union_ty, union_val);
        try sema.storePtr2(block, init_src, union_ptr, init_src, union_init, init_src, .store);
        return;
    }

    try sema.requireRuntimeBlock(block, init_src);
    const new_tag = try sema.addConstant(union_obj.tag_ty, tag_val);
    _ = try block.addBinOp(.set_union_tag, union_ptr, new_tag);
}

fn validateStructInit(
    sema: *Sema,
    block: *Block,
    struct_obj: *Module.Struct,
    init_src: LazySrcLoc,
    instrs: []const Zir.Inst.Index,
    is_comptime: bool,
) CompileError!void {
    const gpa = sema.gpa;

    // Maps field index to field_ptr index of where it was already initialized.
    const found_fields = try gpa.alloc(Zir.Inst.Index, struct_obj.fields.count());
    defer gpa.free(found_fields);
    mem.set(Zir.Inst.Index, found_fields, 0);

    var struct_ptr_zir_ref: Zir.Inst.Ref = undefined;

    for (instrs) |field_ptr| {
        const field_ptr_data = sema.code.instructions.items(.data)[field_ptr].pl_node;
        const field_src: LazySrcLoc = .{ .node_offset_back2tok = field_ptr_data.src_node };
        const field_ptr_extra = sema.code.extraData(Zir.Inst.Field, field_ptr_data.payload_index).data;
        struct_ptr_zir_ref = field_ptr_extra.lhs;
        const field_name = sema.code.nullTerminatedString(field_ptr_extra.field_name_start);
        const field_index = struct_obj.fields.getIndex(field_name) orelse
            return sema.failWithBadStructFieldAccess(block, struct_obj, field_src, field_name);
        if (found_fields[field_index] != 0) {
            const other_field_ptr = found_fields[field_index];
            const other_field_ptr_data = sema.code.instructions.items(.data)[other_field_ptr].pl_node;
            const other_field_src: LazySrcLoc = .{ .node_offset_back2tok = other_field_ptr_data.src_node };
            const msg = msg: {
                const msg = try sema.errMsg(block, field_src, "duplicate field", .{});
                errdefer msg.destroy(gpa);
                try sema.errNote(block, other_field_src, msg, "other field here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
        found_fields[field_index] = field_ptr;
    }

    var root_msg: ?*Module.ErrorMsg = null;

    const fields = struct_obj.fields.values();
    const struct_ptr = sema.resolveInst(struct_ptr_zir_ref);
    const struct_ty = sema.typeOf(struct_ptr).childType();

    if (is_comptime or block.is_comptime) {
        // In this case the only thing we need to do is evaluate the implicit
        // store instructions for default field values, and report any missing fields.
        // Avoid the cost of the extra machinery for detecting a comptime struct init value.
        for (found_fields) |field_ptr, i| {
            if (field_ptr != 0) continue;

            const field = fields[i];
            const field_name = struct_obj.fields.keys()[i];

            if (field.default_val.tag() == .unreachable_value) {
                const template = "missing struct field: {s}";
                const args = .{field_name};
                if (root_msg) |msg| {
                    try sema.errNote(block, init_src, msg, template, args);
                } else {
                    root_msg = try sema.errMsg(block, init_src, template, args);
                }
                continue;
            }

            const default_field_ptr = try sema.structFieldPtr(block, init_src, struct_ptr, field_name, init_src, struct_ty);
            const init = try sema.addConstant(field.ty, field.default_val);
            const field_src = init_src; // TODO better source location
            try sema.storePtr2(block, init_src, default_field_ptr, init_src, init, field_src, .store);
        }

        if (root_msg) |msg| {
            const fqn = try struct_obj.getFullyQualifiedName(gpa);
            defer gpa.free(fqn);
            try sema.mod.errNoteNonLazy(
                struct_obj.srcLoc(),
                msg,
                "struct '{s}' declared here",
                .{fqn},
            );
            return sema.failWithOwnedErrorMsg(block, msg);
        }

        return;
    }

    var struct_is_comptime = true;
    var first_block_index = block.instructions.items.len;

    const air_tags = sema.air_instructions.items(.tag);
    const air_datas = sema.air_instructions.items(.data);

    // We collect the comptime field values in case the struct initialization
    // ends up being comptime-known.
    const field_values = try sema.arena.alloc(Value, fields.len);

    field: for (found_fields) |field_ptr, i| {
        const field = fields[i];

        if (field_ptr != 0) {
            const field_ptr_data = sema.code.instructions.items(.data)[field_ptr].pl_node;
            const field_src: LazySrcLoc = .{ .node_offset_back2tok = field_ptr_data.src_node };

            // Determine whether the value stored to this pointer is comptime-known.
            if (try sema.typeHasOnePossibleValue(block, field_src, field.ty)) |opv| {
                field_values[i] = opv;
                continue;
            }

            const field_ptr_air_ref = sema.inst_map.get(field_ptr).?;
            const field_ptr_air_inst = Air.refToIndex(field_ptr_air_ref).?;

            //std.debug.print("validateStructInit (field_ptr_air_inst=%{d}):\n", .{
            //    field_ptr_air_inst,
            //});
            //for (block.instructions.items) |item| {
            //    std.debug.print("  %{d} = {s}\n", .{item, @tagName(air_tags[item])});
            //}

            // We expect to see something like this in the current block AIR:
            //   %a = field_ptr(...)
            //   store(%a, %b)
            // If %b is a comptime operand, this field is comptime.
            //
            // However, in the case of a comptime-known pointer to a struct, the
            // the field_ptr instruction is missing, so we have to pattern-match
            // based only on the store instructions.
            // `first_block_index` needs to point to the `field_ptr` if it exists;
            // the `store` otherwise.
            //
            // It's also possible for there to be no store instruction, in the case
            // of nested `coerce_result_ptr` instructions. If we see the `field_ptr`
            // but we have not found a `store`, treat as a runtime-known field.

            // Possible performance enhancement: save the `block_index` between iterations
            // of the for loop.
            var block_index = block.instructions.items.len - 1;
            while (block_index > 0) : (block_index -= 1) {
                const store_inst = block.instructions.items[block_index];
                if (store_inst == field_ptr_air_inst) {
                    struct_is_comptime = false;
                    continue :field;
                }
                if (air_tags[store_inst] != .store) continue;
                const bin_op = air_datas[store_inst].bin_op;
                if (bin_op.lhs != field_ptr_air_ref) continue;
                if (block_index > 0 and
                    field_ptr_air_inst == block.instructions.items[block_index - 1])
                {
                    first_block_index = @minimum(first_block_index, block_index - 1);
                } else {
                    first_block_index = @minimum(first_block_index, block_index);
                }
                if (try sema.resolveMaybeUndefValAllowVariables(block, field_src, bin_op.rhs)) |val| {
                    field_values[i] = val;
                } else {
                    struct_is_comptime = false;
                }
                continue :field;
            }
            struct_is_comptime = false;
            continue :field;
        }

        const field_name = struct_obj.fields.keys()[i];

        if (field.default_val.tag() == .unreachable_value) {
            const template = "missing struct field: {s}";
            const args = .{field_name};
            if (root_msg) |msg| {
                try sema.errNote(block, init_src, msg, template, args);
            } else {
                root_msg = try sema.errMsg(block, init_src, template, args);
            }
            continue;
        }
    }

    if (root_msg) |msg| {
        const fqn = try struct_obj.getFullyQualifiedName(gpa);
        defer gpa.free(fqn);
        try sema.mod.errNoteNonLazy(
            struct_obj.srcLoc(),
            msg,
            "struct '{s}' declared here",
            .{fqn},
        );
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    if (struct_is_comptime) {
        // Our task is to delete all the `field_ptr` and `store` instructions, and insert
        // instead a single `store` to the struct_ptr with a comptime struct value.

        block.instructions.shrinkRetainingCapacity(first_block_index);

        // The `field_values` array has been populated for all the non-default struct
        // fields. Here we fill in the default field values.
        for (found_fields) |field_ptr, i| {
            if (field_ptr != 0) continue;

            field_values[i] = fields[i].default_val;
        }

        const struct_val = try Value.Tag.aggregate.create(sema.arena, field_values);
        const struct_init = try sema.addConstant(struct_ty, struct_val);
        try sema.storePtr2(block, init_src, struct_ptr, init_src, struct_init, init_src, .store);
        return;
    }

    // Our task is to insert `store` instructions for all the default field values.

    for (found_fields) |field_ptr, i| {
        if (field_ptr != 0) continue;

        const field = fields[i];
        const field_name = struct_obj.fields.keys()[i];
        const default_field_ptr = try sema.structFieldPtr(block, init_src, struct_ptr, field_name, init_src, struct_ty);

        const init = try sema.addConstant(field.ty, field.default_val);
        const field_src = init_src; // TODO better source location
        try sema.storePtr2(block, init_src, default_field_ptr, init_src, init, field_src, .store);
    }
}

fn zirValidateArrayInit(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_comptime: bool,
) CompileError!void {
    const validate_inst = sema.code.instructions.items(.data)[inst].pl_node;
    const init_src = validate_inst.src();
    const validate_extra = sema.code.extraData(Zir.Inst.Block, validate_inst.payload_index);
    const instrs = sema.code.extra[validate_extra.end..][0..validate_extra.data.body_len];
    const first_elem_ptr_data = sema.code.instructions.items(.data)[instrs[0]].pl_node;
    const elem_ptr_extra = sema.code.extraData(Zir.Inst.ElemPtrImm, first_elem_ptr_data.payload_index).data;
    const array_ptr = sema.resolveInst(elem_ptr_extra.ptr);
    const array_ty = sema.typeOf(array_ptr).childType();
    const array_len = array_ty.arrayLen();

    if (instrs.len != array_len) {
        return sema.fail(block, init_src, "expected {d} array elements; found {d}", .{
            array_len, instrs.len,
        });
    }

    if (is_comptime or block.is_comptime) {
        // In this case the comptime machinery will have evaluated the store instructions
        // at comptime so we have almost nothing to do here. However, in case of a
        // sentinel-terminated array, the sentinel will not have been populated by
        // any ZIR instructions at comptime; we need to do that here.
        if (array_ty.sentinel()) |sentinel_val| {
            const array_len_ref = try sema.addIntUnsigned(Type.usize, array_len);
            const sentinel_ptr = try sema.elemPtrArray(block, init_src, array_ptr, init_src, array_len_ref);
            const sentinel = try sema.addConstant(array_ty.childType(), sentinel_val);
            try sema.storePtr2(block, init_src, sentinel_ptr, init_src, sentinel, init_src, .store);
        }
        return;
    }

    var array_is_comptime = true;
    var first_block_index = block.instructions.items.len;

    // Collect the comptime element values in case the array literal ends up
    // being comptime-known.
    const array_len_s = try sema.usizeCast(block, init_src, array_ty.arrayLenIncludingSentinel());
    const element_vals = try sema.arena.alloc(Value, array_len_s);
    const opt_opv = try sema.typeHasOnePossibleValue(block, init_src, array_ty);
    const air_tags = sema.air_instructions.items(.tag);
    const air_datas = sema.air_instructions.items(.data);

    for (instrs) |elem_ptr, i| {
        const elem_ptr_data = sema.code.instructions.items(.data)[elem_ptr].pl_node;
        const elem_src: LazySrcLoc = .{ .node_offset = elem_ptr_data.src_node };

        // Determine whether the value stored to this pointer is comptime-known.

        const elem_ptr_air_ref = sema.inst_map.get(elem_ptr).?;
        const elem_ptr_air_inst = Air.refToIndex(elem_ptr_air_ref).?;
        // Find the block index of the elem_ptr so that we can look at the next
        // instruction after it within the same block.
        // Possible performance enhancement: save the `block_index` between iterations
        // of the for loop.
        var block_index = block.instructions.items.len - 1;
        while (block.instructions.items[block_index] != elem_ptr_air_inst) {
            block_index -= 1;
        }
        first_block_index = @minimum(first_block_index, block_index);

        // Array has one possible value, so value is always comptime-known
        if (opt_opv) |opv| {
            element_vals[i] = opv;
            continue;
        }

        // If the next instructon is a store with a comptime operand, this element
        // is comptime.
        const next_air_inst = block.instructions.items[block_index + 1];
        switch (air_tags[next_air_inst]) {
            .store => {
                const bin_op = air_datas[next_air_inst].bin_op;
                if (bin_op.lhs != elem_ptr_air_ref) {
                    array_is_comptime = false;
                    continue;
                }
                if (try sema.resolveMaybeUndefValAllowVariables(block, elem_src, bin_op.rhs)) |val| {
                    element_vals[i] = val;
                } else {
                    array_is_comptime = false;
                }
                continue;
            },
            else => {
                array_is_comptime = false;
                continue;
            },
        }
    }

    if (array_is_comptime) {
        // Our task is to delete all the `elem_ptr` and `store` instructions, and insert
        // instead a single `store` to the array_ptr with a comptime struct value.
        // Also to populate the sentinel value, if any.
        if (array_ty.sentinel()) |sentinel_val| {
            element_vals[instrs.len] = sentinel_val;
        }

        block.instructions.shrinkRetainingCapacity(first_block_index);

        const array_val = try Value.Tag.aggregate.create(sema.arena, element_vals);
        const array_init = try sema.addConstant(array_ty, array_val);
        try sema.storePtr2(block, init_src, array_ptr, init_src, array_init, init_src, .store);
    }
}

fn failWithBadMemberAccess(
    sema: *Sema,
    block: *Block,
    agg_ty: Type,
    field_src: LazySrcLoc,
    field_name: []const u8,
) CompileError {
    const kw_name = switch (agg_ty.zigTypeTag()) {
        .Union => "union",
        .Struct => "struct",
        .Opaque => "opaque",
        .Enum => "enum",
        else => unreachable,
    };
    const msg = msg: {
        const target = sema.mod.getTarget();
        const msg = try sema.errMsg(block, field_src, "{s} '{}' has no member named '{s}'", .{
            kw_name, agg_ty.fmt(target), field_name,
        });
        errdefer msg.destroy(sema.gpa);
        try sema.addDeclaredHereNote(msg, agg_ty);
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn failWithBadStructFieldAccess(
    sema: *Sema,
    block: *Block,
    struct_obj: *Module.Struct,
    field_src: LazySrcLoc,
    field_name: []const u8,
) CompileError {
    const gpa = sema.gpa;

    const fqn = try struct_obj.getFullyQualifiedName(gpa);
    defer gpa.free(fqn);

    const msg = msg: {
        const msg = try sema.errMsg(
            block,
            field_src,
            "no field named '{s}' in struct '{s}'",
            .{ field_name, fqn },
        );
        errdefer msg.destroy(gpa);
        try sema.mod.errNoteNonLazy(struct_obj.srcLoc(), msg, "struct declared here", .{});
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn failWithBadUnionFieldAccess(
    sema: *Sema,
    block: *Block,
    union_obj: *Module.Union,
    field_src: LazySrcLoc,
    field_name: []const u8,
) CompileError {
    const gpa = sema.gpa;

    const fqn = try union_obj.getFullyQualifiedName(gpa);
    defer gpa.free(fqn);

    const msg = msg: {
        const msg = try sema.errMsg(
            block,
            field_src,
            "no field named '{s}' in union '{s}'",
            .{ field_name, fqn },
        );
        errdefer msg.destroy(gpa);
        try sema.mod.errNoteNonLazy(union_obj.srcLoc(), msg, "union declared here", .{});
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn addDeclaredHereNote(sema: *Sema, parent: *Module.ErrorMsg, decl_ty: Type) !void {
    const src_loc = decl_ty.declSrcLocOrNull() orelse return;
    const category = switch (decl_ty.zigTypeTag()) {
        .Union => "union",
        .Struct => "struct",
        .Enum => "enum",
        .Opaque => "opaque",
        .ErrorSet => "error set",
        else => unreachable,
    };
    try sema.mod.errNoteNonLazy(src_loc, parent, "{s} declared here", .{category});
}

fn zirStoreToBlockPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const ptr = sema.inst_map.get(Zir.refToIndex(bin_inst.lhs).?) orelse {
        // This is an elided instruction, but AstGen was unable to omit it.
        return;
    };
    const operand = sema.resolveInst(bin_inst.rhs);
    const src: LazySrcLoc = sema.src;
    blk: {
        const ptr_inst = Air.refToIndex(ptr) orelse break :blk;
        if (sema.air_instructions.items(.tag)[ptr_inst] != .constant) break :blk;
        const air_datas = sema.air_instructions.items(.data);
        const ptr_val = sema.air_values.items[air_datas[ptr_inst].ty_pl.payload];
        switch (ptr_val.tag()) {
            .inferred_alloc_comptime => {
                const iac = ptr_val.castTag(.inferred_alloc_comptime).?;
                return sema.storeToInferredAllocComptime(block, src, operand, iac);
            },
            .inferred_alloc => {
                const inferred_alloc = ptr_val.castTag(.inferred_alloc).?;
                return sema.storeToInferredAlloc(block, src, ptr, operand, inferred_alloc);
            },
            else => break :blk,
        }
    }

    return sema.storePtr(block, src, ptr, operand);
}

fn zirStoreToInferredPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = sema.src;
    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const ptr = sema.resolveInst(bin_inst.lhs);
    const operand = sema.resolveInst(bin_inst.rhs);
    const ptr_inst = Air.refToIndex(ptr).?;
    assert(sema.air_instructions.items(.tag)[ptr_inst] == .constant);
    const air_datas = sema.air_instructions.items(.data);
    const ptr_val = sema.air_values.items[air_datas[ptr_inst].ty_pl.payload];

    switch (ptr_val.tag()) {
        .inferred_alloc_comptime => {
            const iac = ptr_val.castTag(.inferred_alloc_comptime).?;
            return sema.storeToInferredAllocComptime(block, src, operand, iac);
        },
        .inferred_alloc => {
            const inferred_alloc = ptr_val.castTag(.inferred_alloc).?;
            return sema.storeToInferredAlloc(block, src, ptr, operand, inferred_alloc);
        },
        else => unreachable,
    }
}

fn storeToInferredAlloc(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr: Air.Inst.Ref,
    operand: Air.Inst.Ref,
    inferred_alloc: *Value.Payload.InferredAlloc,
) CompileError!void {
    const operand_ty = sema.typeOf(operand);
    // Add the stored instruction to the set we will use to resolve peer types
    // for the inferred allocation.
    try inferred_alloc.data.stored_inst_list.append(sema.arena, operand);
    // Create a runtime bitcast instruction with exactly the type the pointer wants.
    const target = sema.mod.getTarget();
    const ptr_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = operand_ty,
        .@"align" = inferred_alloc.data.alignment,
        .@"addrspace" = target_util.defaultAddressSpace(target, .local),
    });
    const bitcasted_ptr = try block.addBitCast(ptr_ty, ptr);
    return sema.storePtr(block, src, bitcasted_ptr, operand);
}

fn storeToInferredAllocComptime(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    operand: Air.Inst.Ref,
    iac: *Value.Payload.InferredAllocComptime,
) CompileError!void {
    const operand_ty = sema.typeOf(operand);
    // There will be only one store_to_inferred_ptr because we are running at comptime.
    // The alloc will turn into a Decl.
    if (try sema.resolveMaybeUndefValAllowVariables(block, src, operand)) |operand_val| {
        if (operand_val.tag() == .variable) {
            return sema.failWithNeededComptime(block, src);
        }
        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();
        iac.data.decl = try anon_decl.finish(
            try operand_ty.copy(anon_decl.arena()),
            try operand_val.copy(anon_decl.arena()),
            iac.data.alignment,
        );
        return;
    } else {
        return sema.failWithNeededComptime(block, src);
    }
}

fn zirSetEvalBranchQuota(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const quota = @intCast(u32, try sema.resolveInt(block, src, inst_data.operand, Type.u32));
    if (sema.branch_quota < quota)
        sema.branch_quota = quota;
}

fn zirStore(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const ptr = sema.resolveInst(bin_inst.lhs);
    const value = sema.resolveInst(bin_inst.rhs);
    return sema.storePtr(block, sema.src, ptr, value);
}

fn zirStoreNode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const zir_tags = sema.code.instructions.items(.tag);
    const zir_datas = sema.code.instructions.items(.data);
    const inst_data = zir_datas[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const ptr = sema.resolveInst(extra.lhs);
    const operand = sema.resolveInst(extra.rhs);

    // Check for the possibility of this pattern:
    //   %a = ret_ptr
    //   %b = store(%a, %c)
    // Where %c is an error union or error set. In such case we need to add
    // to the current function's inferred error set, if any.
    if ((sema.typeOf(operand).zigTypeTag() == .ErrorUnion or
        sema.typeOf(operand).zigTypeTag() == .ErrorSet) and
        sema.fn_ret_ty.zigTypeTag() == .ErrorUnion)
    {
        if (Zir.refToIndex(extra.lhs)) |ptr_index| {
            if (zir_tags[ptr_index] == .extended and
                zir_datas[ptr_index].extended.opcode == .ret_ptr)
            {
                try sema.addToInferredErrorSet(operand);
            }
        }
    }

    return sema.storePtr(block, src, ptr, operand);
}

fn zirParamType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const callee_src = sema.src;

    const inst_data = sema.code.instructions.items(.data)[inst].param_type;
    const callee = sema.resolveInst(inst_data.callee);
    const callee_ty = sema.typeOf(callee);
    var param_index = inst_data.param_index;

    const fn_ty = if (callee_ty.tag() == .bound_fn) fn_ty: {
        const bound_fn_val = try sema.resolveConstValue(block, callee_src, callee);
        const bound_fn = bound_fn_val.castTag(.bound_fn).?.data;
        const fn_ty = sema.typeOf(bound_fn.func_inst);
        param_index += 1;
        break :fn_ty fn_ty;
    } else callee_ty;

    const fn_info = if (fn_ty.zigTypeTag() == .Pointer)
        fn_ty.childType().fnInfo()
    else
        fn_ty.fnInfo();

    if (param_index >= fn_info.param_types.len) {
        if (fn_info.is_var_args) {
            return sema.addType(Type.initTag(.var_args_param));
        }
        // TODO implement begin_call/end_call Zir instructions and check
        // argument count before casting arguments to parameter types.
        return sema.fail(block, callee_src, "wrong number of arguments", .{});
    }

    if (fn_info.param_types[param_index].tag() == .generic_poison) {
        return sema.addType(Type.initTag(.var_args_param));
    }

    return sema.addType(fn_info.param_types[param_index]);
}

fn zirStr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const bytes = sema.code.instructions.items(.data)[inst].str.get(sema.code);
    return sema.addStrLit(block, bytes);
}

fn addStrLit(sema: *Sema, block: *Block, zir_bytes: []const u8) CompileError!Air.Inst.Ref {
    // `zir_bytes` references memory inside the ZIR module, which can get deallocated
    // after semantic analysis is complete, for example in the case of the initialization
    // expression of a variable declaration. We need the memory to be in the new
    // anonymous Decl's arena.
    var anon_decl = try block.startAnonDecl(LazySrcLoc.unneeded);
    defer anon_decl.deinit();

    const bytes = try anon_decl.arena().dupeZ(u8, zir_bytes);

    const new_decl = try anon_decl.finish(
        try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
        try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
        0, // default alignment
    );

    return sema.analyzeDeclRef(new_decl);
}

fn zirInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const tracy = trace(@src());
    defer tracy.end();

    const int = sema.code.instructions.items(.data)[inst].int;
    return sema.addIntUnsigned(Type.initTag(.comptime_int), int);
}

fn zirIntBig(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const tracy = trace(@src());
    defer tracy.end();

    const arena = sema.arena;
    const int = sema.code.instructions.items(.data)[inst].str;
    const byte_count = int.len * @sizeOf(std.math.big.Limb);
    const limb_bytes = sema.code.string_bytes[int.start..][0..byte_count];
    const limbs = try arena.alloc(std.math.big.Limb, int.len);
    mem.copy(u8, mem.sliceAsBytes(limbs), limb_bytes);

    return sema.addConstant(
        Type.initTag(.comptime_int),
        try Value.Tag.int_big_positive.create(arena, limbs),
    );
}

fn zirFloat(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const arena = sema.arena;
    const number = sema.code.instructions.items(.data)[inst].float;
    return sema.addConstant(
        Type.initTag(.comptime_float),
        try Value.Tag.float_64.create(arena, number),
    );
}

fn zirFloat128(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const arena = sema.arena;
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Float128, inst_data.payload_index).data;
    const number = extra.get();
    return sema.addConstant(
        Type.initTag(.comptime_float),
        try Value.Tag.float_128.create(arena, number),
    );
}

fn zirCompileError(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const msg = try sema.resolveConstString(block, operand_src, inst_data.operand);
    return sema.fail(block, src, "{s}", .{msg});
}

fn zirCompileLog(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    var managed = sema.mod.compile_log_text.toManaged(sema.gpa);
    defer sema.mod.compile_log_text = managed.moveToUnmanaged();
    const writer = managed.writer();

    const extra = sema.code.extraData(Zir.Inst.NodeMultiOp, extended.operand);
    const src_node = extra.data.src_node;
    const src: LazySrcLoc = .{ .node_offset = src_node };
    const args = sema.code.refSlice(extra.end, extended.small);
    const target = sema.mod.getTarget();

    for (args) |arg_ref, i| {
        if (i != 0) try writer.print(", ", .{});

        const arg = sema.resolveInst(arg_ref);
        const arg_ty = sema.typeOf(arg);
        if (try sema.resolveMaybeUndefVal(block, src, arg)) |val| {
            try writer.print("@as({}, {})", .{
                arg_ty.fmt(target), val.fmtValue(arg_ty, target),
            });
        } else {
            try writer.print("@as({}, [runtime value])", .{arg_ty.fmt(target)});
        }
    }
    try writer.print("\n", .{});

    const gop = try sema.mod.compile_log_decls.getOrPut(sema.gpa, sema.owner_decl);
    if (!gop.found_existing) {
        gop.value_ptr.* = src_node;
    }
    return Air.Inst.Ref.void_value;
}

fn zirPanic(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src: LazySrcLoc = inst_data.src();
    const msg_inst = sema.resolveInst(inst_data.operand);

    return sema.panicWithMsg(block, src, msg_inst);
}

fn zirLoop(sema: *Sema, parent_block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.Block, inst_data.payload_index);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];
    const gpa = sema.gpa;

    // AIR expects a block outside the loop block too.
    // Reserve space for a Loop instruction so that generated Break instructions can
    // point to it, even if it doesn't end up getting used because the code ends up being
    // comptime evaluated.
    const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
    const loop_inst = block_inst + 1;
    try sema.air_instructions.ensureUnusedCapacity(gpa, 2);
    sema.air_instructions.appendAssumeCapacity(.{
        .tag = .block,
        .data = undefined,
    });
    sema.air_instructions.appendAssumeCapacity(.{
        .tag = .loop,
        .data = .{ .ty_pl = .{
            .ty = .noreturn_type,
            .payload = undefined,
        } },
    });
    var label: Block.Label = .{
        .zir_block = inst,
        .merges = .{
            .results = .{},
            .br_list = .{},
            .block_inst = block_inst,
        },
    };
    var child_block = parent_block.makeSubBlock();
    child_block.label = &label;
    child_block.runtime_cond = null;
    child_block.runtime_loop = src;
    child_block.runtime_index += 1;
    const merges = &child_block.label.?.merges;

    defer child_block.instructions.deinit(gpa);
    defer merges.results.deinit(gpa);
    defer merges.br_list.deinit(gpa);

    var loop_block = child_block.makeSubBlock();
    defer loop_block.instructions.deinit(gpa);

    try sema.analyzeBody(&loop_block, body);

    try child_block.instructions.append(gpa, loop_inst);

    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.Block).Struct.fields.len +
        loop_block.instructions.items.len);
    sema.air_instructions.items(.data)[loop_inst].ty_pl.payload = sema.addExtraAssumeCapacity(
        Air.Block{ .body_len = @intCast(u32, loop_block.instructions.items.len) },
    );
    sema.air_extra.appendSliceAssumeCapacity(loop_block.instructions.items);
    return sema.analyzeBlockBody(parent_block, src, &child_block, merges);
}

fn zirCImport(sema: *Sema, parent_block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const pl_node = sema.code.instructions.items(.data)[inst].pl_node;
    const src = pl_node.src();
    const extra = sema.code.extraData(Zir.Inst.Block, pl_node.payload_index);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];

    // we check this here to avoid undefined symbols
    if (!@import("build_options").have_llvm)
        return sema.fail(parent_block, src, "cannot do C import on Zig compiler not built with LLVM-extension", .{});

    var c_import_buf = std.ArrayList(u8).init(sema.gpa);
    defer c_import_buf.deinit();

    var child_block: Block = .{
        .parent = parent_block,
        .sema = sema,
        .src_decl = parent_block.src_decl,
        .namespace = parent_block.namespace,
        .wip_capture_scope = parent_block.wip_capture_scope,
        .instructions = .{},
        .inlining = parent_block.inlining,
        .is_comptime = parent_block.is_comptime,
        .c_import_buf = &c_import_buf,
    };
    defer child_block.instructions.deinit(sema.gpa);

    // Ignore the result, all the relevant operations have written to c_import_buf already.
    _ = try sema.analyzeBodyBreak(&child_block, body);

    const c_import_res = sema.mod.comp.cImport(c_import_buf.items) catch |err|
        return sema.fail(&child_block, src, "C import failed: {s}", .{@errorName(err)});

    if (c_import_res.errors.len != 0) {
        const msg = msg: {
            const msg = try sema.errMsg(&child_block, src, "C import failed", .{});
            errdefer msg.destroy(sema.gpa);

            if (!sema.mod.comp.bin_file.options.link_libc)
                try sema.errNote(&child_block, src, msg, "libc headers not available; compilation does not link against libc", .{});

            for (c_import_res.errors) |_| {
                // TODO integrate with LazySrcLoc
                // try sema.mod.errNoteNonLazy(.{}, msg, "{s}", .{clang_err.msg_ptr[0..clang_err.msg_len]});
                // if (clang_err.filename_ptr) |p| p[0..clang_err.filename_len] else "(no file)",
                // clang_err.line + 1,
                // clang_err.column + 1,
            }
            @import("clang.zig").Stage2ErrorMsg.delete(c_import_res.errors.ptr, c_import_res.errors.len);
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(parent_block, msg);
    }
    const c_import_pkg = Package.create(
        sema.gpa,
        null,
        c_import_res.out_zig_path,
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => unreachable, // we pass null for root_src_dir_path
    };
    const std_pkg = sema.mod.main_pkg.table.get("std").?;
    const builtin_pkg = sema.mod.main_pkg.table.get("builtin").?;
    try c_import_pkg.add(sema.gpa, "builtin", builtin_pkg);
    try c_import_pkg.add(sema.gpa, "std", std_pkg);

    const result = sema.mod.importPkg(c_import_pkg) catch |err|
        return sema.fail(&child_block, src, "C import failed: {s}", .{@errorName(err)});

    sema.mod.astGenFile(result.file) catch |err|
        return sema.fail(&child_block, src, "C import failed: {s}", .{@errorName(err)});

    try sema.mod.semaFile(result.file);
    const file_root_decl = result.file.root_decl.?;
    try sema.mod.declareDeclDependency(sema.owner_decl, file_root_decl);
    return sema.addConstant(file_root_decl.ty, file_root_decl.val);
}

fn zirSuspendBlock(sema: *Sema, parent_block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    return sema.fail(parent_block, src, "TODO: implement Sema.zirSuspendBlock", .{});
}

fn zirBlock(sema: *Sema, parent_block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const pl_node = sema.code.instructions.items(.data)[inst].pl_node;
    const src = pl_node.src();
    const extra = sema.code.extraData(Zir.Inst.Block, pl_node.payload_index);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];
    const gpa = sema.gpa;

    // Reserve space for a Block instruction so that generated Break instructions can
    // point to it, even if it doesn't end up getting used because the code ends up being
    // comptime evaluated.
    const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
    try sema.air_instructions.append(gpa, .{
        .tag = .block,
        .data = undefined,
    });

    var label: Block.Label = .{
        .zir_block = inst,
        .merges = .{
            .results = .{},
            .br_list = .{},
            .block_inst = block_inst,
        },
    };

    var child_block: Block = .{
        .parent = parent_block,
        .sema = sema,
        .src_decl = parent_block.src_decl,
        .namespace = parent_block.namespace,
        .wip_capture_scope = parent_block.wip_capture_scope,
        .instructions = .{},
        .label = &label,
        .inlining = parent_block.inlining,
        .is_comptime = parent_block.is_comptime,
    };
    const merges = &child_block.label.?.merges;

    defer child_block.instructions.deinit(gpa);
    defer merges.results.deinit(gpa);
    defer merges.br_list.deinit(gpa);

    return sema.resolveBlockBody(parent_block, src, &child_block, body, inst, merges);
}

fn resolveBlockBody(
    sema: *Sema,
    parent_block: *Block,
    src: LazySrcLoc,
    child_block: *Block,
    body: []const Zir.Inst.Index,
    /// This is the instruction that a break instruction within `body` can
    /// use to return from the body.
    body_inst: Zir.Inst.Index,
    merges: *Block.Merges,
) CompileError!Air.Inst.Ref {
    if (child_block.is_comptime) {
        return sema.resolveBody(child_block, body, body_inst);
    } else {
        if (sema.analyzeBodyInner(child_block, body)) |_| {
            return sema.analyzeBlockBody(parent_block, src, child_block, merges);
        } else |err| switch (err) {
            error.ComptimeBreak => {
                const break_inst = sema.comptime_break_inst;
                const break_data = sema.code.instructions.items(.data)[break_inst].@"break";
                if (break_data.block_inst == body_inst) {
                    return sema.resolveInst(break_data.operand);
                } else {
                    return error.ComptimeBreak;
                }
            },
            else => |e| return e,
        }
    }
}

fn analyzeBlockBody(
    sema: *Sema,
    parent_block: *Block,
    src: LazySrcLoc,
    child_block: *Block,
    merges: *Block.Merges,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const gpa = sema.gpa;

    // Blocks must terminate with noreturn instruction.
    assert(child_block.instructions.items.len != 0);
    assert(sema.typeOf(Air.indexToRef(child_block.instructions.items[child_block.instructions.items.len - 1])).isNoReturn());

    if (merges.results.items.len == 0) {
        // No need for a block instruction. We can put the new instructions
        // directly into the parent block.
        try parent_block.instructions.appendSlice(gpa, child_block.instructions.items);
        return Air.indexToRef(child_block.instructions.items[child_block.instructions.items.len - 1]);
    }
    if (merges.results.items.len == 1) {
        const last_inst_index = child_block.instructions.items.len - 1;
        const last_inst = child_block.instructions.items[last_inst_index];
        if (sema.getBreakBlock(last_inst)) |br_block| {
            if (br_block == merges.block_inst) {
                // No need for a block instruction. We can put the new instructions directly
                // into the parent block. Here we omit the break instruction.
                const without_break = child_block.instructions.items[0..last_inst_index];
                try parent_block.instructions.appendSlice(gpa, without_break);
                return merges.results.items[0];
            }
        }
    }
    // It is impossible to have the number of results be > 1 in a comptime scope.
    assert(!child_block.is_comptime); // Should already got a compile error in the condbr condition.

    // Need to set the type and emit the Block instruction. This allows machine code generation
    // to emit a jump instruction to after the block when it encounters the break.
    try parent_block.instructions.append(gpa, merges.block_inst);
    const resolved_ty = try sema.resolvePeerTypes(parent_block, src, merges.results.items, .none);

    const type_src = src; // TODO: better source location
    const valid_rt = try sema.validateRunTimeType(child_block, type_src, resolved_ty, false);
    const target = sema.mod.getTarget();
    if (!valid_rt) {
        const msg = msg: {
            const msg = try sema.errMsg(child_block, type_src, "value with comptime only type '{}' depends on runtime control flow", .{resolved_ty.fmt(target)});
            errdefer msg.destroy(sema.gpa);

            const runtime_src = child_block.runtime_cond orelse child_block.runtime_loop.?;
            try sema.errNote(child_block, runtime_src, msg, "runtime control flow here", .{});

            try sema.explainWhyTypeIsComptime(child_block, type_src, msg, type_src.toSrcLoc(child_block.src_decl), resolved_ty);

            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(child_block, msg);
    }
    const ty_inst = try sema.addType(resolved_ty);
    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.Block).Struct.fields.len +
        child_block.instructions.items.len);
    sema.air_instructions.items(.data)[merges.block_inst] = .{ .ty_pl = .{
        .ty = ty_inst,
        .payload = sema.addExtraAssumeCapacity(Air.Block{
            .body_len = @intCast(u32, child_block.instructions.items.len),
        }),
    } };
    sema.air_extra.appendSliceAssumeCapacity(child_block.instructions.items);
    // Now that the block has its type resolved, we need to go back into all the break
    // instructions, and insert type coercion on the operands.
    for (merges.br_list.items) |br| {
        const br_operand = sema.air_instructions.items(.data)[br].br.operand;
        const br_operand_src = src;
        const br_operand_ty = sema.typeOf(br_operand);
        if (br_operand_ty.eql(resolved_ty, target)) {
            // No type coercion needed.
            continue;
        }
        var coerce_block = parent_block.makeSubBlock();
        defer coerce_block.instructions.deinit(gpa);
        const coerced_operand = try sema.coerce(&coerce_block, resolved_ty, br_operand, br_operand_src);
        // If no instructions were produced, such as in the case of a coercion of a
        // constant value to a new type, we can simply point the br operand to it.
        if (coerce_block.instructions.items.len == 0) {
            sema.air_instructions.items(.data)[br].br.operand = coerced_operand;
            continue;
        }
        assert(coerce_block.instructions.items[coerce_block.instructions.items.len - 1] ==
            Air.refToIndex(coerced_operand).?);

        // Convert the br instruction to a block instruction that has the coercion
        // and then a new br inside that returns the coerced instruction.
        const sub_block_len = @intCast(u32, coerce_block.instructions.items.len + 1);
        try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.Block).Struct.fields.len +
            sub_block_len);
        try sema.air_instructions.ensureUnusedCapacity(gpa, 1);
        const sub_br_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);

        sema.air_instructions.items(.tag)[br] = .block;
        sema.air_instructions.items(.data)[br] = .{ .ty_pl = .{
            .ty = Air.Inst.Ref.noreturn_type,
            .payload = sema.addExtraAssumeCapacity(Air.Block{
                .body_len = sub_block_len,
            }),
        } };
        sema.air_extra.appendSliceAssumeCapacity(coerce_block.instructions.items);
        sema.air_extra.appendAssumeCapacity(sub_br_inst);

        sema.air_instructions.appendAssumeCapacity(.{
            .tag = .br,
            .data = .{ .br = .{
                .block_inst = merges.block_inst,
                .operand = coerced_operand,
            } },
        });
    }
    return Air.indexToRef(merges.block_inst);
}

fn zirExport(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Export, inst_data.payload_index).data;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const options_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const decl_name = sema.code.nullTerminatedString(extra.decl_name);
    if (extra.namespace != .none) {
        return sema.fail(block, src, "TODO: implement exporting with field access", .{});
    }
    const decl = try sema.lookupIdentifier(block, operand_src, decl_name);
    const options = try sema.resolveExportOptions(block, options_src, extra.options);
    try sema.analyzeExport(block, src, options, decl);
}

fn zirExportValue(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.ExportValue, inst_data.payload_index).data;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const options_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand = try sema.resolveInstConst(block, operand_src, extra.operand);
    const options = try sema.resolveExportOptions(block, options_src, extra.options);
    const decl = switch (operand.val.tag()) {
        .function => operand.val.castTag(.function).?.data.owner_decl,
        else => return sema.fail(block, operand_src, "TODO implement exporting arbitrary Value objects", .{}), // TODO put this Value into an anonymous Decl and then export it.
    };
    try sema.analyzeExport(block, src, options, decl);
}

pub fn analyzeExport(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    borrowed_options: std.builtin.ExportOptions,
    exported_decl: *Decl,
) !void {
    const Export = Module.Export;
    const mod = sema.mod;
    const target = mod.getTarget();

    try mod.ensureDeclAnalyzed(exported_decl);
    // TODO run the same checks as we do for C ABI struct fields
    switch (exported_decl.ty.zigTypeTag()) {
        .Fn, .Int, .Enum, .Struct, .Union, .Array, .Float => {},
        else => return sema.fail(block, src, "unable to export type '{}'", .{
            exported_decl.ty.fmt(target),
        }),
    }

    const gpa = mod.gpa;

    try mod.decl_exports.ensureUnusedCapacity(gpa, 1);
    try mod.export_owners.ensureUnusedCapacity(gpa, 1);

    const new_export = try gpa.create(Export);
    errdefer gpa.destroy(new_export);

    const symbol_name = try gpa.dupe(u8, borrowed_options.name);
    errdefer gpa.free(symbol_name);

    const section: ?[]const u8 = if (borrowed_options.section) |s| try gpa.dupe(u8, s) else null;
    errdefer if (section) |s| gpa.free(s);

    const src_decl = block.src_decl;
    const owner_decl = sema.owner_decl;

    log.debug("exporting Decl '{s}' as symbol '{s}' from Decl '{s}'", .{
        exported_decl.name, symbol_name, owner_decl.name,
    });

    new_export.* = .{
        .options = .{
            .name = symbol_name,
            .linkage = borrowed_options.linkage,
            .section = section,
        },
        .src = src,
        .link = switch (mod.comp.bin_file.tag) {
            .coff => .{ .coff = {} },
            .elf => .{ .elf = .{} },
            .macho => .{ .macho = .{} },
            .plan9 => .{ .plan9 = null },
            .c => .{ .c = {} },
            .wasm => .{ .wasm = .{} },
            .spirv => .{ .spirv = {} },
            .nvptx => .{ .nvptx = {} },
        },
        .owner_decl = owner_decl,
        .src_decl = src_decl,
        .exported_decl = exported_decl,
        .status = .in_progress,
    };

    // Add to export_owners table.
    const eo_gop = mod.export_owners.getOrPutAssumeCapacity(owner_decl);
    if (!eo_gop.found_existing) {
        eo_gop.value_ptr.* = &[0]*Export{};
    }
    eo_gop.value_ptr.* = try gpa.realloc(eo_gop.value_ptr.*, eo_gop.value_ptr.len + 1);
    eo_gop.value_ptr.*[eo_gop.value_ptr.len - 1] = new_export;
    errdefer eo_gop.value_ptr.* = gpa.shrink(eo_gop.value_ptr.*, eo_gop.value_ptr.len - 1);

    // Add to exported_decl table.
    const de_gop = mod.decl_exports.getOrPutAssumeCapacity(exported_decl);
    if (!de_gop.found_existing) {
        de_gop.value_ptr.* = &[0]*Export{};
    }
    de_gop.value_ptr.* = try gpa.realloc(de_gop.value_ptr.*, de_gop.value_ptr.len + 1);
    de_gop.value_ptr.*[de_gop.value_ptr.len - 1] = new_export;
    errdefer de_gop.value_ptr.* = gpa.shrink(de_gop.value_ptr.*, de_gop.value_ptr.len - 1);
}

fn zirSetAlignStack(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const src: LazySrcLoc = inst_data.src();
    const alignment = try sema.resolveAlign(block, operand_src, inst_data.operand);
    if (alignment > 256) {
        return sema.fail(block, src, "attempt to @setAlignStack({d}); maximum is 256", .{
            alignment,
        });
    }
    const func = sema.owner_func orelse
        return sema.fail(block, src, "@setAlignStack outside function body", .{});

    switch (func.owner_decl.ty.fnCallingConvention()) {
        .Naked => return sema.fail(block, src, "@setAlignStack in naked function", .{}),
        .Inline => return sema.fail(block, src, "@setAlignStack in inline function", .{}),
        else => {},
    }

    const gop = try sema.mod.align_stack_fns.getOrPut(sema.mod.gpa, func);
    if (gop.found_existing) {
        const msg = msg: {
            const msg = try sema.errMsg(block, src, "multiple @setAlignStack in the same function body", .{});
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(block, src, msg, "other instance here", .{});
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }
    gop.value_ptr.* = .{ .alignment = alignment, .src = src };
}

fn zirSetCold(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const is_cold = try sema.resolveConstBool(block, operand_src, inst_data.operand);
    const func = sema.func orelse return; // does nothing outside a function
    func.is_cold = is_cold;
}

fn zirSetFloatMode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src: LazySrcLoc = inst_data.src();
    const float_mode = try sema.resolveBuiltinEnum(block, src, inst_data.operand, "FloatMode");
    switch (float_mode) {
        .Strict => return,
        .Optimized => {
            // TODO implement optimized float mode
        },
    }
}

fn zirSetRuntimeSafety(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    block.want_safety = try sema.resolveConstBool(block, operand_src, inst_data.operand);
}

fn zirFence(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    if (block.is_comptime) return;

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const order_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const order = try sema.resolveAtomicOrder(block, order_src, inst_data.operand);

    if (@enumToInt(order) < @enumToInt(std.builtin.AtomicOrder.Acquire)) {
        return sema.fail(block, order_src, "atomic ordering must be Acquire or stricter", .{});
    }

    _ = try block.addInst(.{
        .tag = .fence,
        .data = .{ .fence = order },
    });
}

fn zirBreak(sema: *Sema, start_block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].@"break";
    const operand = sema.resolveInst(inst_data.operand);
    const zir_block = inst_data.block_inst;

    var block = start_block;
    while (true) {
        if (block.label) |label| {
            if (label.zir_block == zir_block) {
                const br_ref = try start_block.addBr(label.merges.block_inst, operand);
                try label.merges.results.append(sema.gpa, operand);
                try label.merges.br_list.append(sema.gpa, Air.refToIndex(br_ref).?);
                block.runtime_index += 1;
                if (block.runtime_cond == null and block.runtime_loop == null) {
                    block.runtime_cond = start_block.runtime_cond orelse start_block.runtime_loop;
                    block.runtime_loop = start_block.runtime_loop;
                }
                return inst;
            }
        }
        block = block.parent.?;
    }
}

fn zirDbgStmt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    // We do not set sema.src here because dbg_stmt instructions are only emitted for
    // ZIR code that possibly will need to generate runtime code. So error messages
    // and other source locations must not rely on sema.src being set from dbg_stmt
    // instructions.
    if (block.is_comptime or sema.mod.comp.bin_file.options.strip) return;

    const inst_data = sema.code.instructions.items(.data)[inst].dbg_stmt;
    _ = try block.addInst(.{
        .tag = .dbg_stmt,
        .data = .{ .dbg_stmt = .{
            .line = inst_data.line,
            .column = inst_data.column,
        } },
    });
}

fn zirDbgBlockBegin(sema: *Sema, block: *Block) CompileError!void {
    if (block.is_comptime or sema.mod.comp.bin_file.options.strip) return;

    _ = try block.addInst(.{
        .tag = .dbg_block_begin,
        .data = undefined,
    });
}

fn zirDbgBlockEnd(sema: *Sema, block: *Block) CompileError!void {
    if (block.is_comptime or sema.mod.comp.bin_file.options.strip) return;

    _ = try block.addInst(.{
        .tag = .dbg_block_end,
        .data = undefined,
    });
}

fn zirDbgVar(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!void {
    if (block.is_comptime or sema.mod.comp.bin_file.options.strip) return;

    const str_op = sema.code.instructions.items(.data)[inst].str_op;
    const operand = sema.resolveInst(str_op.operand);
    const name = str_op.getStr(sema.code);
    try sema.addDbgVar(block, operand, air_tag, name);
}

fn addDbgVar(
    sema: *Sema,
    block: *Block,
    operand: Air.Inst.Ref,
    air_tag: Air.Inst.Tag,
    name: []const u8,
) CompileError!void {
    const operand_ty = sema.typeOf(operand);
    switch (air_tag) {
        .dbg_var_ptr => {
            if (!(try sema.typeHasRuntimeBits(block, sema.src, operand_ty.childType()))) return;
        },
        .dbg_var_val => {
            if (!(try sema.typeHasRuntimeBits(block, sema.src, operand_ty))) return;
        },
        else => unreachable,
    }

    try sema.queueFullTypeResolution(operand_ty);

    // Add the name to the AIR.
    const name_extra_index = @intCast(u32, sema.air_extra.items.len);
    const elements_used = name.len / 4 + 1;
    try sema.air_extra.ensureUnusedCapacity(sema.gpa, elements_used);
    const buffer = mem.sliceAsBytes(sema.air_extra.unusedCapacitySlice());
    mem.copy(u8, buffer, name);
    buffer[name.len] = 0;
    sema.air_extra.items.len += elements_used;

    _ = try block.addInst(.{
        .tag = air_tag,
        .data = .{ .pl_op = .{
            .payload = name_extra_index,
            .operand = operand,
        } },
    });
}

fn zirDeclRef(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const src = inst_data.src();
    const decl_name = inst_data.get(sema.code);
    const decl = try sema.lookupIdentifier(block, src, decl_name);
    return sema.analyzeDeclRef(decl);
}

fn zirDeclVal(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const src = inst_data.src();
    const decl_name = inst_data.get(sema.code);
    const decl = try sema.lookupIdentifier(block, src, decl_name);
    return sema.analyzeDeclVal(block, src, decl);
}

fn lookupIdentifier(sema: *Sema, block: *Block, src: LazySrcLoc, name: []const u8) !*Decl {
    var namespace = block.namespace;
    while (true) {
        if (try sema.lookupInNamespace(block, src, namespace, name, false)) |decl| {
            return decl;
        }
        namespace = namespace.parent orelse break;
    }
    unreachable; // AstGen detects use of undeclared identifier errors.
}

/// This looks up a member of a specific namespace. It is affected by `usingnamespace` but
/// only for ones in the specified namespace.
fn lookupInNamespace(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    namespace: *Namespace,
    ident_name: []const u8,
    observe_usingnamespace: bool,
) CompileError!?*Decl {
    const mod = sema.mod;

    const namespace_decl = namespace.getDecl();
    if (namespace_decl.analysis == .file_failure) {
        try mod.declareDeclDependency(sema.owner_decl, namespace_decl);
        return error.AnalysisFail;
    }

    if (observe_usingnamespace and namespace.usingnamespace_set.count() != 0) {
        const src_file = block.namespace.file_scope;

        const gpa = sema.gpa;
        var checked_namespaces: std.AutoArrayHashMapUnmanaged(*Namespace, void) = .{};
        defer checked_namespaces.deinit(gpa);

        // Keep track of name conflicts for error notes.
        var candidates: std.ArrayListUnmanaged(*Decl) = .{};
        defer candidates.deinit(gpa);

        try checked_namespaces.put(gpa, namespace, {});
        var check_i: usize = 0;

        while (check_i < checked_namespaces.count()) : (check_i += 1) {
            const check_ns = checked_namespaces.keys()[check_i];
            if (check_ns.decls.getKeyAdapted(ident_name, Module.DeclAdapter{})) |decl| {
                // Skip decls which are not marked pub, which are in a different
                // file than the `a.b`/`@hasDecl` syntax.
                if (decl.is_pub or src_file == decl.getFileScope()) {
                    try candidates.append(gpa, decl);
                }
            }
            var it = check_ns.usingnamespace_set.iterator();
            while (it.next()) |entry| {
                const sub_usingnamespace_decl = entry.key_ptr.*;
                const sub_is_pub = entry.value_ptr.*;
                if (!sub_is_pub and src_file != sub_usingnamespace_decl.getFileScope()) {
                    // Skip usingnamespace decls which are not marked pub, which are in
                    // a different file than the `a.b`/`@hasDecl` syntax.
                    continue;
                }
                try sema.ensureDeclAnalyzed(sub_usingnamespace_decl);
                const ns_ty = sub_usingnamespace_decl.val.castTag(.ty).?.data;
                const sub_ns = ns_ty.getNamespace().?;
                try checked_namespaces.put(gpa, sub_ns, {});
            }
        }

        switch (candidates.items.len) {
            0 => {},
            1 => {
                const decl = candidates.items[0];
                try mod.declareDeclDependency(sema.owner_decl, decl);
                return decl;
            },
            else => {
                const msg = msg: {
                    const msg = try sema.errMsg(block, src, "ambiguous reference", .{});
                    errdefer msg.destroy(gpa);
                    for (candidates.items) |candidate| {
                        const src_loc = candidate.srcLoc();
                        try mod.errNoteNonLazy(src_loc, msg, "declared here", .{});
                    }
                    break :msg msg;
                };
                return sema.failWithOwnedErrorMsg(block, msg);
            },
        }
    } else if (namespace.decls.getKeyAdapted(ident_name, Module.DeclAdapter{})) |decl| {
        try mod.declareDeclDependency(sema.owner_decl, decl);
        return decl;
    }

    log.debug("{*} ({s}) depends on non-existence of '{s}' in {*} ({s})", .{
        sema.owner_decl, sema.owner_decl.name, ident_name, namespace_decl, namespace_decl.name,
    });
    // TODO This dependency is too strong. Really, it should only be a dependency
    // on the non-existence of `ident_name` in the namespace. We can lessen the number of
    // outdated declarations by making this dependency more sophisticated.
    try mod.declareDeclDependency(sema.owner_decl, namespace_decl);
    return null;
}

fn zirCall(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const func_src: LazySrcLoc = .{ .node_offset_call_func = inst_data.src_node };
    const call_src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.Call, inst_data.payload_index);
    const args = sema.code.refSlice(extra.end, extra.data.flags.args_len);

    const modifier = @intToEnum(std.builtin.CallOptions.Modifier, extra.data.flags.packed_modifier);
    const ensure_result_used = extra.data.flags.ensure_result_used;

    var func = sema.resolveInst(extra.data.callee);
    var resolved_args: []Air.Inst.Ref = undefined;

    const func_type = sema.typeOf(func);

    // Desugar bound functions here
    if (func_type.tag() == .bound_fn) {
        const bound_func = try sema.resolveValue(block, func_src, func);
        const bound_data = &bound_func.cast(Value.Payload.BoundFn).?.data;
        func = bound_data.func_inst;
        resolved_args = try sema.arena.alloc(Air.Inst.Ref, args.len + 1);
        resolved_args[0] = bound_data.arg0_inst;
        for (args) |zir_arg, i| {
            resolved_args[i + 1] = sema.resolveInst(zir_arg);
        }
    } else {
        resolved_args = try sema.arena.alloc(Air.Inst.Ref, args.len);
        for (args) |zir_arg, i| {
            resolved_args[i] = sema.resolveInst(zir_arg);
        }
    }

    return sema.analyzeCall(block, func, func_src, call_src, modifier, ensure_result_used, resolved_args);
}

const GenericCallAdapter = struct {
    generic_fn: *Module.Fn,
    precomputed_hash: u64,
    func_ty_info: Type.Payload.Function.Data,
    comptime_tvs: []const TypedValue,
    target: std.Target,

    pub fn eql(ctx: @This(), adapted_key: void, other_key: *Module.Fn) bool {
        _ = adapted_key;
        // The generic function Decl is guaranteed to be the first dependency
        // of each of its instantiations.
        const generic_owner_decl = other_key.owner_decl.dependencies.keys()[0];
        if (ctx.generic_fn.owner_decl != generic_owner_decl) return false;

        const other_comptime_args = other_key.comptime_args.?;
        for (other_comptime_args[0..ctx.func_ty_info.param_types.len]) |other_arg, i| {
            if (other_arg.ty.tag() != .generic_poison) {
                // anytype parameter
                if (!other_arg.ty.eql(ctx.comptime_tvs[i].ty, ctx.target)) {
                    return false;
                }
            }
            if (other_arg.val.tag() != .generic_poison) {
                // comptime parameter
                if (ctx.comptime_tvs[i].val.tag() == .generic_poison) {
                    // No match because the instantiation has a comptime parameter
                    // but the callsite does not.
                    return false;
                }
                if (!other_arg.val.eql(ctx.comptime_tvs[i].val, other_arg.ty, ctx.target)) {
                    return false;
                }
            }
        }
        return true;
    }

    /// The implementation of the hash is in semantic analysis of function calls, so
    /// that any errors when computing the hash can be properly reported.
    pub fn hash(ctx: @This(), adapted_key: void) u64 {
        _ = adapted_key;
        return ctx.precomputed_hash;
    }
};

fn analyzeCall(
    sema: *Sema,
    block: *Block,
    func: Air.Inst.Ref,
    func_src: LazySrcLoc,
    call_src: LazySrcLoc,
    modifier: std.builtin.CallOptions.Modifier,
    ensure_result_used: bool,
    uncasted_args: []const Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const mod = sema.mod;

    const callee_ty = sema.typeOf(func);
    const target = sema.mod.getTarget();
    const func_ty = func_ty: {
        switch (callee_ty.zigTypeTag()) {
            .Fn => break :func_ty callee_ty,
            .Pointer => {
                const ptr_info = callee_ty.ptrInfo().data;
                if (ptr_info.size == .One and ptr_info.pointee_type.zigTypeTag() == .Fn) {
                    break :func_ty ptr_info.pointee_type;
                }
            },
            else => {},
        }
        return sema.fail(block, func_src, "type '{}' not a function", .{callee_ty.fmt(target)});
    };

    const func_ty_info = func_ty.fnInfo();
    const cc = func_ty_info.cc;
    if (cc == .Naked) {
        // TODO add error note: declared here
        return sema.fail(
            block,
            func_src,
            "unable to call function with naked calling convention",
            .{},
        );
    }
    const fn_params_len = func_ty_info.param_types.len;
    if (func_ty_info.is_var_args) {
        assert(cc == .C);
        if (uncasted_args.len < fn_params_len) {
            // TODO add error note: declared here
            return sema.fail(
                block,
                func_src,
                "expected at least {d} argument(s), found {d}",
                .{ fn_params_len, uncasted_args.len },
            );
        }
    } else if (fn_params_len != uncasted_args.len) {
        // TODO add error note: declared here
        return sema.fail(
            block,
            func_src,
            "expected {d} argument(s), found {d}",
            .{ fn_params_len, uncasted_args.len },
        );
    }

    const call_tag: Air.Inst.Tag = switch (modifier) {
        .auto,
        .always_inline,
        .compile_time,
        .no_async,
        => Air.Inst.Tag.call,

        .never_tail => Air.Inst.Tag.call_never_tail,
        .never_inline => Air.Inst.Tag.call_never_inline,
        .always_tail => Air.Inst.Tag.call_always_tail,

        .async_kw => return sema.fail(block, call_src, "TODO implement async call", .{}),
    };

    const gpa = sema.gpa;

    var is_generic_call = func_ty_info.is_generic;
    var is_comptime_call = block.is_comptime or modifier == .compile_time;
    if (!is_comptime_call) {
        if (sema.typeRequiresComptime(block, func_src, func_ty_info.return_type)) |ct| {
            is_comptime_call = ct;
        } else |err| switch (err) {
            error.GenericPoison => is_generic_call = true,
            else => |e| return e,
        }
    }
    var is_inline_call = is_comptime_call or modifier == .always_inline or
        func_ty_info.cc == .Inline;

    if (!is_inline_call and is_generic_call) {
        if (sema.instantiateGenericCall(
            block,
            func,
            func_src,
            call_src,
            func_ty_info,
            ensure_result_used,
            uncasted_args,
            call_tag,
        )) |some| {
            return some;
        } else |err| switch (err) {
            error.GenericPoison => {
                is_inline_call = true;
            },
            error.ComptimeReturn => {
                is_inline_call = true;
                is_comptime_call = true;
            },
            else => |e| return e,
        }
    }

    const result: Air.Inst.Ref = if (is_inline_call) res: {
        const func_val = try sema.resolveConstValue(block, func_src, func);
        const module_fn = switch (func_val.tag()) {
            .decl_ref => func_val.castTag(.decl_ref).?.data.val.castTag(.function).?.data,
            .function => func_val.castTag(.function).?.data,
            .extern_fn => return sema.fail(block, call_src, "{s} call of extern function", .{
                @as([]const u8, if (is_comptime_call) "comptime" else "inline"),
            }),
            else => unreachable,
        };

        // Analyze the ZIR. The same ZIR gets analyzed into a runtime function
        // or an inlined call depending on what union tag the `label` field is
        // set to in the `Block`.
        // This block instruction will be used to capture the return value from the
        // inlined function.
        const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
        try sema.air_instructions.append(gpa, .{
            .tag = .block,
            .data = undefined,
        });
        // This one is shared among sub-blocks within the same callee, but not
        // shared among the entire inline/comptime call stack.
        var inlining: Block.Inlining = .{
            .comptime_result = undefined,
            .merges = .{
                .results = .{},
                .br_list = .{},
                .block_inst = block_inst,
            },
        };
        // In order to save a bit of stack space, directly modify Sema rather
        // than create a child one.
        const parent_zir = sema.code;
        sema.code = module_fn.owner_decl.getFileScope().zir;
        defer sema.code = parent_zir;

        const parent_inst_map = sema.inst_map;
        sema.inst_map = .{};
        defer {
            sema.inst_map.deinit(gpa);
            sema.inst_map = parent_inst_map;
        }

        const parent_func = sema.func;
        sema.func = module_fn;
        defer sema.func = parent_func;

        var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, module_fn.owner_decl.src_scope);
        defer wip_captures.deinit();

        var child_block: Block = .{
            .parent = null,
            .sema = sema,
            .src_decl = module_fn.owner_decl,
            .namespace = module_fn.owner_decl.src_namespace,
            .wip_capture_scope = wip_captures.scope,
            .instructions = .{},
            .label = null,
            .inlining = &inlining,
            .is_comptime = is_comptime_call,
        };

        const merges = &child_block.inlining.?.merges;

        defer child_block.instructions.deinit(gpa);
        defer merges.results.deinit(gpa);
        defer merges.br_list.deinit(gpa);

        // If it's a comptime function call, we need to memoize it as long as no external
        // comptime memory is mutated.
        var memoized_call_key: Module.MemoizedCall.Key = undefined;
        var delete_memoized_call_key = false;
        defer if (delete_memoized_call_key) gpa.free(memoized_call_key.args);
        if (is_comptime_call) {
            memoized_call_key = .{
                .func = module_fn,
                .args = try gpa.alloc(TypedValue, func_ty_info.param_types.len),
            };
            delete_memoized_call_key = true;
        }

        try sema.emitBackwardBranch(&child_block, call_src);

        // Whether this call should be memoized, set to false if the call can mutate
        // comptime state.
        var should_memoize = true;

        var new_fn_info = module_fn.owner_decl.ty.fnInfo();
        new_fn_info.param_types = try sema.arena.alloc(Type, new_fn_info.param_types.len);
        new_fn_info.comptime_params = (try sema.arena.alloc(bool, new_fn_info.param_types.len)).ptr;

        // This will have return instructions analyzed as break instructions to
        // the block_inst above. Here we are performing "comptime/inline semantic analysis"
        // for a function body, which means we must map the parameter ZIR instructions to
        // the AIR instructions of the callsite. The callee could be a generic function
        // which means its parameter type expressions must be resolved in order and used
        // to successively coerce the arguments.
        const fn_info = sema.code.getFnInfo(module_fn.zir_body_inst);
        const zir_tags = sema.code.instructions.items(.tag);
        var arg_i: usize = 0;
        for (fn_info.param_body) |inst| switch (zir_tags[inst]) {
            .param, .param_comptime => {
                // Evaluate the parameter type expression now that previous ones have
                // been mapped, and coerce the corresponding argument to it.
                const pl_tok = sema.code.instructions.items(.data)[inst].pl_tok;
                const param_src = pl_tok.src();
                const extra = sema.code.extraData(Zir.Inst.Param, pl_tok.payload_index);
                const param_body = sema.code.extra[extra.end..][0..extra.data.body_len];
                const param_ty_inst = try sema.resolveBody(&child_block, param_body, inst);
                const param_ty = try sema.analyzeAsType(&child_block, param_src, param_ty_inst);
                new_fn_info.param_types[arg_i] = param_ty;
                const arg_src = call_src; // TODO: better source location
                const casted_arg = try sema.coerce(&child_block, param_ty, uncasted_args[arg_i], arg_src);
                try sema.inst_map.putNoClobber(gpa, inst, casted_arg);

                if (is_comptime_call) {
                    const arg_val = try sema.resolveConstMaybeUndefVal(&child_block, arg_src, casted_arg);
                    switch (arg_val.tag()) {
                        .generic_poison, .generic_poison_type => {
                            // This function is currently evaluated as part of an as-of-yet unresolvable
                            // parameter or return type.
                            return error.GenericPoison;
                        },
                        else => {},
                    }
                    should_memoize = should_memoize and !arg_val.canMutateComptimeVarState();
                    memoized_call_key.args[arg_i] = .{
                        .ty = param_ty,
                        .val = arg_val,
                    };
                }

                arg_i += 1;
                continue;
            },
            .param_anytype, .param_anytype_comptime => {
                // No coercion needed.
                const uncasted_arg = uncasted_args[arg_i];
                new_fn_info.param_types[arg_i] = sema.typeOf(uncasted_arg);
                try sema.inst_map.putNoClobber(gpa, inst, uncasted_arg);

                if (is_comptime_call) {
                    const arg_src = call_src; // TODO: better source location
                    const arg_val = try sema.resolveConstMaybeUndefVal(&child_block, arg_src, uncasted_arg);
                    switch (arg_val.tag()) {
                        .generic_poison, .generic_poison_type => {
                            // This function is currently evaluated as part of an as-of-yet unresolvable
                            // parameter or return type.
                            return error.GenericPoison;
                        },
                        else => {},
                    }
                    should_memoize = should_memoize and !arg_val.canMutateComptimeVarState();
                    memoized_call_key.args[arg_i] = .{
                        .ty = sema.typeOf(uncasted_arg),
                        .val = arg_val,
                    };
                }

                arg_i += 1;
                continue;
            },
            else => continue,
        };

        // In case it is a generic function with an expression for the return type that depends
        // on parameters, we must now do the same for the return type as we just did with
        // each of the parameters, resolving the return type and providing it to the child
        // `Sema` so that it can be used for the `ret_ptr` instruction.
        const ret_ty_inst = try sema.resolveBody(&child_block, fn_info.ret_ty_body, module_fn.zir_body_inst);
        const ret_ty_src = func_src; // TODO better source location
        const bare_return_type = try sema.analyzeAsType(&child_block, ret_ty_src, ret_ty_inst);
        // Create a fresh inferred error set type for inline/comptime calls.
        const fn_ret_ty = blk: {
            if (module_fn.hasInferredErrorSet()) {
                const node = try sema.gpa.create(Module.Fn.InferredErrorSetListNode);
                node.data = .{ .func = module_fn };
                if (parent_func) |some| {
                    some.inferred_error_sets.prepend(node);
                }

                const error_set_ty = try Type.Tag.error_set_inferred.create(sema.arena, &node.data);
                break :blk try Type.Tag.error_union.create(sema.arena, .{
                    .error_set = error_set_ty,
                    .payload = bare_return_type,
                });
            }
            break :blk bare_return_type;
        };
        new_fn_info.return_type = fn_ret_ty;
        const parent_fn_ret_ty = sema.fn_ret_ty;
        sema.fn_ret_ty = fn_ret_ty;
        defer sema.fn_ret_ty = parent_fn_ret_ty;

        // This `res2` is here instead of directly breaking from `res` due to a stage1
        // bug generating invalid LLVM IR.
        const res2: Air.Inst.Ref = res2: {
            if (should_memoize and is_comptime_call) {
                if (mod.memoized_calls.getContext(memoized_call_key, .{ .target = target })) |result| {
                    const ty_inst = try sema.addType(fn_ret_ty);
                    try sema.air_values.append(gpa, result.val);
                    sema.air_instructions.set(block_inst, .{
                        .tag = .constant,
                        .data = .{ .ty_pl = .{
                            .ty = ty_inst,
                            .payload = @intCast(u32, sema.air_values.items.len - 1),
                        } },
                    });
                    break :res2 Air.indexToRef(block_inst);
                }
            }

            const new_func_resolved_ty = try Type.Tag.function.create(sema.arena, new_fn_info);
            if (!is_comptime_call) {
                try sema.emitDbgInline(block, parent_func.?, module_fn, new_func_resolved_ty, .dbg_inline_begin);

                for (fn_info.param_body) |param| switch (zir_tags[param]) {
                    .param, .param_comptime => {
                        const inst_data = sema.code.instructions.items(.data)[param].pl_tok;
                        const extra = sema.code.extraData(Zir.Inst.Param, inst_data.payload_index);
                        const param_name = sema.code.nullTerminatedString(extra.data.name);
                        const inst = sema.inst_map.get(param).?;

                        try sema.addDbgVar(&child_block, inst, .dbg_var_val, param_name);
                    },
                    .param_anytype, .param_anytype_comptime => {
                        const inst_data = sema.code.instructions.items(.data)[param].str_tok;
                        const param_name = inst_data.get(sema.code);
                        const inst = sema.inst_map.get(param).?;

                        try sema.addDbgVar(&child_block, inst, .dbg_var_val, param_name);
                    },
                    else => continue,
                };
            }

            const result = result: {
                sema.analyzeBody(&child_block, fn_info.body) catch |err| switch (err) {
                    error.ComptimeReturn => break :result inlining.comptime_result,
                    error.AnalysisFail => {
                        const err_msg = inlining.err orelse return err;
                        try sema.errNote(block, call_src, err_msg, "called from here", .{});
                        if (block.inlining) |some| some.err = err_msg;
                        return err;
                    },
                    else => |e| return e,
                };
                break :result try sema.analyzeBlockBody(block, call_src, &child_block, merges);
            };

            if (!is_comptime_call) {
                try sema.emitDbgInline(block, module_fn, parent_func.?, parent_func.?.owner_decl.ty, .dbg_inline_end);
            }

            if (should_memoize and is_comptime_call) {
                const result_val = try sema.resolveConstMaybeUndefVal(block, call_src, result);

                // TODO: check whether any external comptime memory was mutated by the
                // comptime function call. If so, then do not memoize the call here.
                // TODO: re-evaluate whether memoized_calls needs its own arena. I think
                // it should be fine to use the Decl arena for the function.
                {
                    var arena_allocator = std.heap.ArenaAllocator.init(gpa);
                    errdefer arena_allocator.deinit();
                    const arena = arena_allocator.allocator();

                    for (memoized_call_key.args) |*arg| {
                        arg.* = try arg.*.copy(arena);
                    }

                    try mod.memoized_calls.putContext(gpa, memoized_call_key, .{
                        .val = try result_val.copy(arena),
                        .arena = arena_allocator.state,
                    }, .{ .target = sema.mod.getTarget() });
                    delete_memoized_call_key = false;
                }
            }

            break :res2 result;
        };

        try wip_captures.finalize();

        break :res res2;
    } else res: {
        assert(!func_ty_info.is_generic);
        try sema.requireRuntimeBlock(block, call_src);

        const args = try sema.arena.alloc(Air.Inst.Ref, uncasted_args.len);
        for (uncasted_args) |uncasted_arg, i| {
            const arg_src = call_src; // TODO: better source location
            if (i < fn_params_len) {
                const param_ty = func_ty.fnParamType(i);
                try sema.resolveTypeFully(block, arg_src, param_ty);
                args[i] = try sema.coerce(block, param_ty, uncasted_arg, arg_src);
            } else {
                args[i] = uncasted_arg;
            }
        }

        try sema.queueFullTypeResolution(func_ty_info.return_type);

        try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.Call).Struct.fields.len +
            args.len);
        const func_inst = try block.addInst(.{
            .tag = call_tag,
            .data = .{ .pl_op = .{
                .operand = func,
                .payload = sema.addExtraAssumeCapacity(Air.Call{
                    .args_len = @intCast(u32, args.len),
                }),
            } },
        });
        sema.appendRefsAssumeCapacity(args);
        break :res func_inst;
    };

    if (ensure_result_used) {
        try sema.ensureResultUsed(block, result, call_src);
    }
    return result;
}

fn instantiateGenericCall(
    sema: *Sema,
    block: *Block,
    func: Air.Inst.Ref,
    func_src: LazySrcLoc,
    call_src: LazySrcLoc,
    func_ty_info: Type.Payload.Function.Data,
    ensure_result_used: bool,
    uncasted_args: []const Air.Inst.Ref,
    call_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const mod = sema.mod;
    const gpa = sema.gpa;

    const func_val = try sema.resolveConstValue(block, func_src, func);
    const module_fn = switch (func_val.tag()) {
        .function => func_val.castTag(.function).?.data,
        .decl_ref => func_val.castTag(.decl_ref).?.data.val.castTag(.function).?.data,
        else => unreachable,
    };
    // Check the Module's generic function map with an adapted context, so that we
    // can match against `uncasted_args` rather than doing the work below to create a
    // generic Scope only to junk it if it matches an existing instantiation.
    const namespace = module_fn.owner_decl.src_namespace;
    const fn_zir = namespace.file_scope.zir;
    const fn_info = fn_zir.getFnInfo(module_fn.zir_body_inst);
    const zir_tags = fn_zir.instructions.items(.tag);

    // This hash must match `Module.MonomorphedFuncsContext.hash`.
    // For parameters explicitly marked comptime and simple parameter type expressions,
    // we know whether a parameter is elided from a monomorphed function, and can
    // use it in the hash here. However, for parameter type expressions that are not
    // explicitly marked comptime and rely on previous parameter comptime values, we
    // don't find out until after generating a monomorphed function whether the parameter
    // type ended up being a "must-be-comptime-known" type.
    var hasher = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hasher, @ptrToInt(module_fn));

    const comptime_tvs = try sema.arena.alloc(TypedValue, func_ty_info.param_types.len);
    const target = sema.mod.getTarget();

    for (func_ty_info.param_types) |param_ty, i| {
        const is_comptime = func_ty_info.paramIsComptime(i);
        if (is_comptime) {
            const arg_src = call_src; // TODO better source location
            const casted_arg = try sema.coerce(block, param_ty, uncasted_args[i], arg_src);
            if (try sema.resolveMaybeUndefVal(block, arg_src, casted_arg)) |arg_val| {
                if (param_ty.tag() != .generic_poison) {
                    arg_val.hash(param_ty, &hasher, target);
                }
                comptime_tvs[i] = .{
                    // This will be different than `param_ty` in the case of `generic_poison`.
                    .ty = sema.typeOf(casted_arg),
                    .val = arg_val,
                };
            } else {
                return sema.failWithNeededComptime(block, arg_src);
            }
        } else {
            comptime_tvs[i] = .{
                .ty = sema.typeOf(uncasted_args[i]),
                .val = Value.initTag(.generic_poison),
            };
        }
    }

    const precomputed_hash = hasher.final();

    const adapter: GenericCallAdapter = .{
        .generic_fn = module_fn,
        .precomputed_hash = precomputed_hash,
        .func_ty_info = func_ty_info,
        .comptime_tvs = comptime_tvs,
        .target = target,
    };
    const gop = try mod.monomorphed_funcs.getOrPutAdapted(gpa, {}, adapter);
    const callee = if (!gop.found_existing) callee: {
        const new_module_func = try gpa.create(Module.Fn);
        // This ensures that we can operate on the hash map before the Module.Fn
        // struct is fully initialized.
        new_module_func.hash = precomputed_hash;
        gop.key_ptr.* = new_module_func;
        errdefer gpa.destroy(new_module_func);
        errdefer assert(mod.monomorphed_funcs.remove(new_module_func));

        try namespace.anon_decls.ensureUnusedCapacity(gpa, 1);

        // Create a Decl for the new function.
        const src_decl = namespace.getDecl();
        // TODO better names for generic function instantiations
        const name_index = mod.getNextAnonNameIndex();
        const decl_name = try std.fmt.allocPrintZ(gpa, "{s}__anon_{d}", .{
            module_fn.owner_decl.name, name_index,
        });
        const new_decl = try mod.allocateNewDecl(decl_name, namespace, module_fn.owner_decl.src_node, src_decl.src_scope);
        errdefer new_decl.destroy(mod);
        new_decl.src_line = module_fn.owner_decl.src_line;
        new_decl.is_pub = module_fn.owner_decl.is_pub;
        new_decl.is_exported = module_fn.owner_decl.is_exported;
        new_decl.has_align = module_fn.owner_decl.has_align;
        new_decl.has_linksection_or_addrspace = module_fn.owner_decl.has_linksection_or_addrspace;
        new_decl.@"addrspace" = module_fn.owner_decl.@"addrspace";
        new_decl.zir_decl_index = module_fn.owner_decl.zir_decl_index;
        new_decl.alive = true; // This Decl is called at runtime.
        new_decl.analysis = .in_progress;
        new_decl.generation = mod.generation;

        namespace.anon_decls.putAssumeCapacityNoClobber(new_decl, {});
        errdefer assert(namespace.anon_decls.orderedRemove(new_decl));

        // The generic function Decl is guaranteed to be the first dependency
        // of each of its instantiations.
        assert(new_decl.dependencies.keys().len == 0);
        try mod.declareDeclDependency(new_decl, module_fn.owner_decl);
        // Resolving the new function type below will possibly declare more decl dependencies
        // and so we remove them all here in case of error.
        errdefer {
            for (new_decl.dependencies.keys()) |dep| {
                dep.removeDependant(new_decl);
            }
        }

        var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
        errdefer new_decl_arena.deinit();
        const new_decl_arena_allocator = new_decl_arena.allocator();

        // Re-run the block that creates the function, with the comptime parameters
        // pre-populated inside `inst_map`. This causes `param_comptime` and
        // `param_anytype_comptime` ZIR instructions to be ignored, resulting in a
        // new, monomorphized function, with the comptime parameters elided.
        var child_sema: Sema = .{
            .mod = mod,
            .gpa = gpa,
            .arena = sema.arena,
            .perm_arena = new_decl_arena_allocator,
            .code = fn_zir,
            .owner_decl = new_decl,
            .func = null,
            .fn_ret_ty = Type.void,
            .owner_func = null,
            .comptime_args = try new_decl_arena_allocator.alloc(TypedValue, uncasted_args.len),
            .comptime_args_fn_inst = module_fn.zir_body_inst,
            .preallocated_new_func = new_module_func,
        };
        defer child_sema.deinit();

        var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, new_decl.src_scope);
        defer wip_captures.deinit();

        var child_block: Block = .{
            .parent = null,
            .sema = &child_sema,
            .src_decl = new_decl,
            .namespace = namespace,
            .wip_capture_scope = wip_captures.scope,
            .instructions = .{},
            .inlining = null,
            .is_comptime = true,
        };
        defer {
            child_block.instructions.deinit(gpa);
            child_block.params.deinit(gpa);
        }

        try child_sema.inst_map.ensureUnusedCapacity(gpa, @intCast(u32, uncasted_args.len));
        var arg_i: usize = 0;
        for (fn_info.param_body) |inst| {
            var is_comptime = false;
            var is_anytype = false;
            switch (zir_tags[inst]) {
                .param => {
                    is_comptime = func_ty_info.paramIsComptime(arg_i);
                },
                .param_comptime => {
                    is_comptime = true;
                },
                .param_anytype => {
                    is_anytype = true;
                    is_comptime = func_ty_info.paramIsComptime(arg_i);
                },
                .param_anytype_comptime => {
                    is_anytype = true;
                    is_comptime = true;
                },
                else => continue,
            }
            const arg_src = call_src; // TODO: better source location
            const arg = uncasted_args[arg_i];
            if (is_comptime) {
                if (try sema.resolveMaybeUndefVal(block, arg_src, arg)) |arg_val| {
                    const child_arg = try child_sema.addConstant(sema.typeOf(arg), arg_val);
                    child_sema.inst_map.putAssumeCapacityNoClobber(inst, child_arg);
                } else {
                    return sema.failWithNeededComptime(block, arg_src);
                }
            } else if (is_anytype) {
                const arg_ty = sema.typeOf(arg);
                if (try sema.typeRequiresComptime(block, arg_src, arg_ty)) {
                    const arg_val = try sema.resolveConstValue(block, arg_src, arg);
                    const child_arg = try child_sema.addConstant(arg_ty, arg_val);
                    child_sema.inst_map.putAssumeCapacityNoClobber(inst, child_arg);
                } else {
                    // We insert into the map an instruction which is runtime-known
                    // but has the type of the argument.
                    const child_arg = try child_block.addArg(arg_ty);
                    child_sema.inst_map.putAssumeCapacityNoClobber(inst, child_arg);
                }
            }
            arg_i += 1;
        }
        const new_func_inst = child_sema.resolveBody(&child_block, fn_info.param_body, fn_info.param_body_inst) catch |err| {
            // TODO look up the compile error that happened here and attach a note to it
            // pointing here, at the generic instantiation callsite.
            if (sema.owner_func) |owner_func| {
                owner_func.state = .dependency_failure;
            } else {
                sema.owner_decl.analysis = .dependency_failure;
            }
            return err;
        };
        const new_func_val = child_sema.resolveConstValue(&child_block, .unneeded, new_func_inst) catch unreachable;
        const new_func = new_func_val.castTag(.function).?.data;
        errdefer new_func.deinit(gpa);
        assert(new_func == new_module_func);

        arg_i = 0;
        for (fn_info.param_body) |inst| {
            switch (zir_tags[inst]) {
                .param_comptime, .param_anytype_comptime, .param, .param_anytype => {},
                else => continue,
            }
            const arg = child_sema.inst_map.get(inst).?;
            const copied_arg_ty = try child_sema.typeOf(arg).copy(new_decl_arena_allocator);
            if (child_sema.resolveMaybeUndefValAllowVariables(
                &child_block,
                .unneeded,
                arg,
            ) catch unreachable) |arg_val| {
                child_sema.comptime_args[arg_i] = .{
                    .ty = copied_arg_ty,
                    .val = try arg_val.copy(new_decl_arena_allocator),
                };
            } else {
                child_sema.comptime_args[arg_i] = .{
                    .ty = copied_arg_ty,
                    .val = Value.initTag(.generic_poison),
                };
            }

            arg_i += 1;
        }

        try wip_captures.finalize();

        // Populate the Decl ty/val with the function and its type.
        new_decl.ty = try child_sema.typeOf(new_func_inst).copy(new_decl_arena_allocator);
        // If the call evaluated to a return type that requires comptime, never mind
        // our generic instantiation. Instead we need to perform a comptime call.
        const new_fn_info = new_decl.ty.fnInfo();
        if (try sema.typeRequiresComptime(block, call_src, new_fn_info.return_type)) {
            return error.ComptimeReturn;
        }
        // Similarly, if the call evaluated to a generic type we need to instead
        // call it inline.
        if (new_fn_info.is_generic or new_fn_info.cc == .Inline) {
            return error.GenericPoison;
        }

        new_decl.val = try Value.Tag.function.create(new_decl_arena_allocator, new_func);
        new_decl.has_tv = true;
        new_decl.owns_tv = true;
        new_decl.analysis = .complete;

        log.debug("generic function '{s}' instantiated with type {}", .{
            new_decl.name, new_decl.ty.fmtDebug(),
        });

        // Queue up a `codegen_func` work item for the new Fn. The `comptime_args` field
        // will be populated, ensuring it will have `analyzeBody` called with the ZIR
        // parameters mapped appropriately.
        try mod.comp.bin_file.allocateDeclIndexes(new_decl);
        try mod.comp.work_queue.writeItem(.{ .codegen_func = new_func });

        try new_decl.finalizeNewArena(&new_decl_arena);
        break :callee new_func;
    } else gop.key_ptr.*;

    const callee_inst = try sema.analyzeDeclVal(block, func_src, callee.owner_decl);

    // Make a runtime call to the new function, making sure to omit the comptime args.
    try sema.requireRuntimeBlock(block, call_src);

    const comptime_args = callee.comptime_args.?;
    const new_fn_info = callee.owner_decl.ty.fnInfo();
    const runtime_args_len = count: {
        var count: u32 = 0;
        var arg_i: usize = 0;
        for (fn_info.param_body) |inst| {
            switch (zir_tags[inst]) {
                .param_comptime, .param_anytype_comptime, .param, .param_anytype => {
                    if (comptime_args[arg_i].val.tag() == .generic_poison) {
                        count += 1;
                    }
                    arg_i += 1;
                },
                else => continue,
            }
        }
        break :count count;
    };
    const runtime_args = try sema.arena.alloc(Air.Inst.Ref, runtime_args_len);
    {
        var runtime_i: u32 = 0;
        var total_i: u32 = 0;
        for (fn_info.param_body) |inst| {
            switch (zir_tags[inst]) {
                .param_comptime, .param_anytype_comptime, .param, .param_anytype => {},
                else => continue,
            }
            const is_runtime = comptime_args[total_i].val.tag() == .generic_poison;
            if (is_runtime) {
                const param_ty = new_fn_info.param_types[runtime_i];
                const arg_src = call_src; // TODO: better source location
                const uncasted_arg = uncasted_args[total_i];
                const casted_arg = try sema.coerce(block, param_ty, uncasted_arg, arg_src);
                try sema.queueFullTypeResolution(param_ty);
                runtime_args[runtime_i] = casted_arg;
                runtime_i += 1;
            }
            total_i += 1;
        }

        try sema.queueFullTypeResolution(new_fn_info.return_type);
    }
    try sema.air_extra.ensureUnusedCapacity(sema.gpa, @typeInfo(Air.Call).Struct.fields.len +
        runtime_args_len);
    const func_inst = try block.addInst(.{
        .tag = call_tag,
        .data = .{ .pl_op = .{
            .operand = callee_inst,
            .payload = sema.addExtraAssumeCapacity(Air.Call{
                .args_len = runtime_args_len,
            }),
        } },
    });
    sema.appendRefsAssumeCapacity(runtime_args);

    if (ensure_result_used) {
        try sema.ensureResultUsed(block, func_inst, call_src);
    }
    return func_inst;
}

fn emitDbgInline(
    sema: *Sema,
    block: *Block,
    old_func: *Module.Fn,
    new_func: *Module.Fn,
    new_func_ty: Type,
    tag: Air.Inst.Tag,
) CompileError!void {
    if (sema.mod.comp.bin_file.options.strip) return;

    // Recursive inline call; no dbg_inline needed.
    if (old_func == new_func) return;

    try sema.air_values.append(sema.gpa, try Value.Tag.function.create(sema.arena, new_func));
    _ = try block.addInst(.{
        .tag = tag,
        .data = .{ .ty_pl = .{
            .ty = try sema.addType(new_func_ty),
            .payload = @intCast(u32, sema.air_values.items.len - 1),
        } },
    });
}

fn zirIntType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const tracy = trace(@src());
    defer tracy.end();

    const int_type = sema.code.instructions.items(.data)[inst].int_type;
    const ty = try Module.makeIntType(sema.arena, int_type.signedness, int_type.bit_count);

    return sema.addType(ty);
}

fn zirOptionalType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const child_type = try sema.resolveType(block, src, inst_data.operand);
    const opt_type = try Type.optional(sema.arena, child_type);

    return sema.addType(opt_type);
}

fn zirElemType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const array_type = try sema.resolveType(block, src, inst_data.operand);
    const elem_type = array_type.elemType();
    return sema.addType(elem_type);
}

fn zirVectorType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const elem_type_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const len_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const len = try sema.resolveInt(block, len_src, extra.lhs, Type.u32);
    const elem_type = try sema.resolveType(block, elem_type_src, extra.rhs);
    try sema.checkVectorElemType(block, elem_type_src, elem_type);
    const vector_type = try Type.Tag.vector.create(sema.arena, .{
        .len = @intCast(u32, len),
        .elem_type = elem_type,
    });
    return sema.addType(vector_type);
}

fn zirArrayType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const len = try sema.resolveInt(block, .unneeded, bin_inst.lhs, Type.usize);
    const elem_type = try sema.resolveType(block, .unneeded, bin_inst.rhs);
    const target = sema.mod.getTarget();
    const array_ty = try Type.array(sema.arena, len, null, elem_type, target);

    return sema.addType(array_ty);
}

fn zirArrayTypeSentinel(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.ArrayTypeSentinel, inst_data.payload_index).data;
    const len_src: LazySrcLoc = .{ .node_offset_array_type_len = inst_data.src_node };
    const sentinel_src: LazySrcLoc = .{ .node_offset_array_type_sentinel = inst_data.src_node };
    const elem_src: LazySrcLoc = .{ .node_offset_array_type_elem = inst_data.src_node };
    const len = try sema.resolveInt(block, len_src, extra.len, Type.usize);
    const elem_type = try sema.resolveType(block, elem_src, extra.elem_type);
    const uncasted_sentinel = sema.resolveInst(extra.sentinel);
    const sentinel = try sema.coerce(block, elem_type, uncasted_sentinel, sentinel_src);
    const sentinel_val = try sema.resolveConstValue(block, sentinel_src, sentinel);
    const target = sema.mod.getTarget();
    const array_ty = try Type.array(sema.arena, len, sentinel_val, elem_type, target);

    return sema.addType(array_ty);
}

fn zirAnyframeType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_anyframe_type = inst_data.src_node };
    const return_type = try sema.resolveType(block, operand_src, inst_data.operand);
    const anyframe_type = try Type.Tag.anyframe_T.create(sema.arena, return_type);

    return sema.addType(anyframe_type);
}

fn zirErrorUnionType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const error_set = try sema.resolveType(block, lhs_src, extra.lhs);
    const payload = try sema.resolveType(block, rhs_src, extra.rhs);
    const target = sema.mod.getTarget();

    if (error_set.zigTypeTag() != .ErrorSet) {
        return sema.fail(block, lhs_src, "expected error set type, found {}", .{
            error_set.fmt(target),
        });
    }
    const err_union_ty = try Type.errorUnion(sema.arena, error_set, payload, target);
    return sema.addType(err_union_ty);
}

fn zirErrorValue(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;

    // Create an anonymous error set type with only this error value, and return the value.
    const kv = try sema.mod.getErrorValue(inst_data.get(sema.code));
    const result_type = try Type.Tag.error_set_single.create(sema.arena, kv.key);
    return sema.addConstant(
        result_type,
        try Value.Tag.@"error".create(sema.arena, .{
            .name = kv.key,
        }),
    );
}

fn zirErrorToInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const op = sema.resolveInst(inst_data.operand);
    const op_coerced = try sema.coerce(block, Type.anyerror, op, operand_src);
    const result_ty = Type.u16;

    if (try sema.resolveMaybeUndefVal(block, src, op_coerced)) |val| {
        if (val.isUndef()) {
            return sema.addConstUndef(result_ty);
        }
        const payload = try sema.arena.create(Value.Payload.U64);
        payload.* = .{
            .base = .{ .tag = .int_u64 },
            .data = (try sema.mod.getErrorValue(val.castTag(.@"error").?.data.name)).value,
        };
        return sema.addConstant(result_ty, Value.initPayload(&payload.base));
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addBitCast(result_ty, op_coerced);
}

fn zirIntToError(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const uncasted_operand = sema.resolveInst(inst_data.operand);
    const operand = try sema.coerce(block, Type.u16, uncasted_operand, operand_src);
    const target = sema.mod.getTarget();

    if (try sema.resolveDefinedValue(block, operand_src, operand)) |value| {
        const int = try sema.usizeCast(block, operand_src, value.toUnsignedInt(target));
        if (int > sema.mod.global_error_set.count() or int == 0)
            return sema.fail(block, operand_src, "integer value {d} represents no error", .{int});
        const payload = try sema.arena.create(Value.Payload.Error);
        payload.* = .{
            .base = .{ .tag = .@"error" },
            .data = .{ .name = sema.mod.error_name_list.items[int] },
        };
        return sema.addConstant(Type.anyerror, Value.initPayload(&payload.base));
    }
    try sema.requireRuntimeBlock(block, src);
    if (block.wantSafety()) {
        const is_lt_len = try block.addUnOp(.cmp_lt_errors_len, operand);
        try sema.addSafetyCheck(block, is_lt_len, .invalid_error_code);
    }
    return block.addInst(.{
        .tag = .bitcast,
        .data = .{ .ty_op = .{
            .ty = Air.Inst.Ref.anyerror_type,
            .operand = operand,
        } },
    });
}

fn zirMergeErrorSets(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src: LazySrcLoc = .{ .node_offset_bin_op = inst_data.src_node };
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    if (sema.typeOf(lhs).zigTypeTag() == .Bool and sema.typeOf(rhs).zigTypeTag() == .Bool) {
        const msg = msg: {
            const msg = try sema.errMsg(block, lhs_src, "expected error set type, found 'bool'", .{});
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(block, src, msg, "'||' merges error sets; 'or' performs boolean OR", .{});
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }
    const lhs_ty = try sema.analyzeAsType(block, lhs_src, lhs);
    const rhs_ty = try sema.analyzeAsType(block, rhs_src, rhs);
    const target = sema.mod.getTarget();
    if (lhs_ty.zigTypeTag() != .ErrorSet)
        return sema.fail(block, lhs_src, "expected error set type, found {}", .{lhs_ty.fmt(target)});
    if (rhs_ty.zigTypeTag() != .ErrorSet)
        return sema.fail(block, rhs_src, "expected error set type, found {}", .{rhs_ty.fmt(target)});

    // Anything merged with anyerror is anyerror.
    if (lhs_ty.tag() == .anyerror or rhs_ty.tag() == .anyerror) {
        return Air.Inst.Ref.anyerror_type;
    }

    if (lhs_ty.castTag(.error_set_inferred)) |payload| {
        try sema.resolveInferredErrorSet(block, src, payload.data);
        // isAnyError might have changed from a false negative to a true positive after resolution.
        if (lhs_ty.isAnyError()) {
            return Air.Inst.Ref.anyerror_type;
        }
    }
    if (rhs_ty.castTag(.error_set_inferred)) |payload| {
        try sema.resolveInferredErrorSet(block, src, payload.data);
        // isAnyError might have changed from a false negative to a true positive after resolution.
        if (rhs_ty.isAnyError()) {
            return Air.Inst.Ref.anyerror_type;
        }
    }

    const err_set_ty = try lhs_ty.errorSetMerge(sema.arena, rhs_ty);
    return sema.addType(err_set_ty);
}

fn zirEnumLiteral(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const duped_name = try sema.arena.dupe(u8, inst_data.get(sema.code));
    return sema.addConstant(
        Type.initTag(.enum_literal),
        try Value.Tag.enum_literal.create(sema.arena, duped_name),
    );
}

fn zirEnumToInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const arena = sema.arena;
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    const enum_tag: Air.Inst.Ref = switch (operand_ty.zigTypeTag()) {
        .Enum => operand,
        .Union => blk: {
            const tag_ty = operand_ty.unionTagType() orelse {
                return sema.fail(
                    block,
                    operand_src,
                    "untagged union '{}' cannot be converted to integer",
                    .{src},
                );
            };
            break :blk try sema.unionToTag(block, tag_ty, operand, operand_src);
        },
        else => {
            return sema.fail(block, operand_src, "expected enum or tagged union, found {}", .{
                operand_ty.fmt(target),
            });
        },
    };
    const enum_tag_ty = sema.typeOf(enum_tag);

    var int_tag_type_buffer: Type.Payload.Bits = undefined;
    const int_tag_ty = try enum_tag_ty.intTagType(&int_tag_type_buffer).copy(arena);

    if (try sema.typeHasOnePossibleValue(block, src, enum_tag_ty)) |opv| {
        return sema.addConstant(int_tag_ty, opv);
    }

    if (try sema.resolveMaybeUndefVal(block, operand_src, enum_tag)) |enum_tag_val| {
        var buffer: Value.Payload.U64 = undefined;
        const val = enum_tag_val.enumToInt(enum_tag_ty, &buffer);
        return sema.addConstant(int_tag_ty, try val.copy(sema.arena));
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addBitCast(int_tag_ty, enum_tag);
}

fn zirIntToEnum(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const target = sema.mod.getTarget();
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src = inst_data.src();
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);

    if (dest_ty.zigTypeTag() != .Enum) {
        return sema.fail(block, dest_ty_src, "expected enum, found {}", .{dest_ty.fmt(target)});
    }

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |int_val| {
        if (dest_ty.isNonexhaustiveEnum()) {
            return sema.addConstant(dest_ty, int_val);
        }
        if (int_val.isUndef()) {
            return sema.failWithUseOfUndef(block, operand_src);
        }
        if (!dest_ty.enumHasInt(int_val, target)) {
            const msg = msg: {
                const msg = try sema.errMsg(
                    block,
                    src,
                    "enum '{}' has no tag with value {}",
                    .{ dest_ty.fmt(target), int_val.fmtValue(sema.typeOf(operand), target) },
                );
                errdefer msg.destroy(sema.gpa);
                try sema.mod.errNoteNonLazy(
                    dest_ty.declSrcLoc(),
                    msg,
                    "enum declared here",
                    .{},
                );
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
        return sema.addConstant(dest_ty, int_val);
    }

    try sema.requireRuntimeBlock(block, src);
    // TODO insert safety check to make sure the value matches an enum value
    return block.addTyOp(.intcast, dest_ty, operand);
}

/// Pointer in, pointer out.
fn zirOptionalPayloadPtr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    safety_check: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const optional_ptr = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();

    return sema.analyzeOptionalPayloadPtr(block, src, optional_ptr, safety_check, false);
}

fn analyzeOptionalPayloadPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    optional_ptr: Air.Inst.Ref,
    safety_check: bool,
    initializing: bool,
) CompileError!Air.Inst.Ref {
    const optional_ptr_ty = sema.typeOf(optional_ptr);
    assert(optional_ptr_ty.zigTypeTag() == .Pointer);

    const target = sema.mod.getTarget();
    const opt_type = optional_ptr_ty.elemType();
    if (opt_type.zigTypeTag() != .Optional) {
        return sema.fail(block, src, "expected optional type, found {}", .{opt_type.fmt(target)});
    }

    const child_type = try opt_type.optionalChildAlloc(sema.arena);
    const child_pointer = try Type.ptr(sema.arena, target, .{
        .pointee_type = child_type,
        .mutable = !optional_ptr_ty.isConstPtr(),
        .@"addrspace" = optional_ptr_ty.ptrAddressSpace(),
    });

    if (try sema.resolveDefinedValue(block, src, optional_ptr)) |ptr_val| {
        if (initializing) {
            if (!ptr_val.isComptimeMutablePtr()) {
                // If the pointer resulting from this function was stored at comptime,
                // the optional non-null bit would be set that way. But in this case,
                // we need to emit a runtime instruction to do it.
                try sema.requireRuntimeBlock(block, src);
                _ = try block.addTyOp(.optional_payload_ptr_set, child_pointer, optional_ptr);
            }
            return sema.addConstant(
                child_pointer,
                try Value.Tag.opt_payload_ptr.create(sema.arena, .{
                    .container_ptr = ptr_val,
                    .container_ty = optional_ptr_ty.childType(),
                }),
            );
        }
        if (try sema.pointerDeref(block, src, ptr_val, optional_ptr_ty)) |val| {
            if (val.isNull()) {
                return sema.fail(block, src, "unable to unwrap null", .{});
            }
            // The same Value represents the pointer to the optional and the payload.
            return sema.addConstant(
                child_pointer,
                try Value.Tag.opt_payload_ptr.create(sema.arena, .{
                    .container_ptr = ptr_val,
                    .container_ty = optional_ptr_ty.childType(),
                }),
            );
        }
    }

    try sema.requireRuntimeBlock(block, src);
    if (safety_check and block.wantSafety()) {
        const is_non_null = try block.addUnOp(.is_non_null_ptr, optional_ptr);
        try sema.addSafetyCheck(block, is_non_null, .unwrap_null);
    }
    const air_tag: Air.Inst.Tag = if (initializing)
        .optional_payload_ptr_set
    else
        .optional_payload_ptr;
    return block.addTyOp(air_tag, child_pointer, optional_ptr);
}

/// Value in, value out.
fn zirOptionalPayload(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    safety_check: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const result_ty = switch (operand_ty.zigTypeTag()) {
        .Optional => try operand_ty.optionalChildAlloc(sema.arena),
        .Pointer => t: {
            if (operand_ty.ptrSize() != .C) {
                return sema.failWithExpectedOptionalType(block, src, operand_ty);
            }
            const ptr_info = operand_ty.ptrInfo().data;
            const target = sema.mod.getTarget();
            break :t try Type.ptr(sema.arena, target, .{
                .pointee_type = try ptr_info.pointee_type.copy(sema.arena),
                .@"align" = ptr_info.@"align",
                .@"addrspace" = ptr_info.@"addrspace",
                .mutable = ptr_info.mutable,
                .@"allowzero" = ptr_info.@"allowzero",
                .@"volatile" = ptr_info.@"volatile",
                .size = .One,
            });
        },
        else => return sema.failWithExpectedOptionalType(block, src, operand_ty),
    };

    if (try sema.resolveDefinedValue(block, src, operand)) |val| {
        if (val.isNull()) {
            return sema.fail(block, src, "unable to unwrap null", .{});
        }
        if (val.castTag(.opt_payload)) |payload| {
            return sema.addConstant(result_ty, payload.data);
        }
        return sema.addConstant(result_ty, val);
    }

    try sema.requireRuntimeBlock(block, src);
    if (safety_check and block.wantSafety()) {
        const is_non_null = try block.addUnOp(.is_non_null, operand);
        try sema.addSafetyCheck(block, is_non_null, .unwrap_null);
    }
    return block.addTyOp(.optional_payload, result_ty, operand);
}

/// Value in, value out
fn zirErrUnionPayload(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    safety_check: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_src = src;
    const operand_ty = sema.typeOf(operand);
    if (operand_ty.zigTypeTag() != .ErrorUnion) {
        const target = sema.mod.getTarget();
        return sema.fail(block, operand_src, "expected error union type, found '{}'", .{
            operand_ty.fmt(target),
        });
    }

    if (try sema.resolveDefinedValue(block, src, operand)) |val| {
        if (val.getError()) |name| {
            return sema.fail(block, src, "caught unexpected error '{s}'", .{name});
        }
        const data = val.castTag(.eu_payload).?.data;
        const result_ty = operand_ty.errorUnionPayload();
        return sema.addConstant(result_ty, data);
    }
    try sema.requireRuntimeBlock(block, src);
    if (safety_check and block.wantSafety()) {
        const is_non_err = try block.addUnOp(.is_err, operand);
        try sema.addSafetyCheck(block, is_non_err, .unwrap_errunion);
    }
    const result_ty = operand_ty.errorUnionPayload();
    return block.addTyOp(.unwrap_errunion_payload, result_ty, operand);
}

/// Pointer in, pointer out.
fn zirErrUnionPayloadPtr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    safety_check: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();

    return sema.analyzeErrUnionPayloadPtr(block, src, operand, safety_check, false);
}

fn analyzeErrUnionPayloadPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    operand: Air.Inst.Ref,
    safety_check: bool,
    initializing: bool,
) CompileError!Air.Inst.Ref {
    const operand_ty = sema.typeOf(operand);
    assert(operand_ty.zigTypeTag() == .Pointer);

    const target = sema.mod.getTarget();
    if (operand_ty.elemType().zigTypeTag() != .ErrorUnion) {
        return sema.fail(block, src, "expected error union type, found {}", .{
            operand_ty.elemType().fmt(target),
        });
    }

    const payload_ty = operand_ty.elemType().errorUnionPayload();
    const operand_pointer_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = payload_ty,
        .mutable = !operand_ty.isConstPtr(),
        .@"addrspace" = operand_ty.ptrAddressSpace(),
    });

    if (try sema.resolveDefinedValue(block, src, operand)) |ptr_val| {
        if (initializing) {
            if (!ptr_val.isComptimeMutablePtr()) {
                // If the pointer resulting from this function was stored at comptime,
                // the error union error code would be set that way. But in this case,
                // we need to emit a runtime instruction to do it.
                try sema.requireRuntimeBlock(block, src);
                _ = try block.addTyOp(.errunion_payload_ptr_set, operand_pointer_ty, operand);
            }
            return sema.addConstant(
                operand_pointer_ty,
                try Value.Tag.eu_payload_ptr.create(sema.arena, .{
                    .container_ptr = ptr_val,
                    .container_ty = operand_ty.elemType(),
                }),
            );
        }
        if (try sema.pointerDeref(block, src, ptr_val, operand_ty)) |val| {
            if (val.getError()) |name| {
                return sema.fail(block, src, "caught unexpected error '{s}'", .{name});
            }

            return sema.addConstant(
                operand_pointer_ty,
                try Value.Tag.eu_payload_ptr.create(sema.arena, .{
                    .container_ptr = ptr_val,
                    .container_ty = operand_ty.elemType(),
                }),
            );
        }
    }

    try sema.requireRuntimeBlock(block, src);
    if (safety_check and block.wantSafety()) {
        const is_non_err = try block.addUnOp(.is_err, operand);
        try sema.addSafetyCheck(block, is_non_err, .unwrap_errunion);
    }
    const air_tag: Air.Inst.Tag = if (initializing)
        .errunion_payload_ptr_set
    else
        .unwrap_errunion_payload_ptr;
    return block.addTyOp(air_tag, operand_pointer_ty, operand);
}

/// Value in, value out
fn zirErrUnionCode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();
    if (operand_ty.zigTypeTag() != .ErrorUnion) {
        return sema.fail(block, src, "expected error union type, found '{}'", .{
            operand_ty.fmt(target),
        });
    }

    const result_ty = operand_ty.errorUnionSet();

    if (try sema.resolveDefinedValue(block, src, operand)) |val| {
        assert(val.getError() != null);
        return sema.addConstant(result_ty, val);
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.unwrap_errunion_err, result_ty, operand);
}

/// Pointer in, value out
fn zirErrUnionCodePtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    assert(operand_ty.zigTypeTag() == .Pointer);

    if (operand_ty.elemType().zigTypeTag() != .ErrorUnion) {
        const target = sema.mod.getTarget();
        return sema.fail(block, src, "expected error union type, found {}", .{
            operand_ty.elemType().fmt(target),
        });
    }

    const result_ty = operand_ty.elemType().errorUnionSet();

    if (try sema.resolveDefinedValue(block, src, operand)) |pointer_val| {
        if (try sema.pointerDeref(block, src, pointer_val, operand_ty)) |val| {
            assert(val.getError() != null);
            return sema.addConstant(result_ty, val);
        }
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.unwrap_errunion_err_ptr, result_ty, operand);
}

fn zirEnsureErrPayloadVoid(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_tok;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();
    if (operand_ty.zigTypeTag() != .ErrorUnion) {
        return sema.fail(block, src, "expected error union type, found '{}'", .{
            operand_ty.fmt(target),
        });
    }
    if (operand_ty.errorUnionPayload().zigTypeTag() != .Void) {
        return sema.fail(block, src, "expression value is ignored", .{});
    }
}

fn zirFunc(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    inferred_error_set: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Func, inst_data.payload_index);
    var extra_index = extra.end;
    const ret_ty_body = sema.code.extra[extra_index..][0..extra.data.ret_body_len];
    extra_index += ret_ty_body.len;

    var src_locs: Zir.Inst.Func.SrcLocs = undefined;
    const has_body = extra.data.body_len != 0;
    if (has_body) {
        extra_index += extra.data.body_len;
        src_locs = sema.code.extraData(Zir.Inst.Func.SrcLocs, extra_index).data;
    }

    const cc: std.builtin.CallingConvention = if (sema.owner_decl.is_exported)
        .C
    else
        .Unspecified;

    return sema.funcCommon(
        block,
        inst_data.src_node,
        inst,
        ret_ty_body,
        cc,
        Value.@"null",
        false,
        inferred_error_set,
        false,
        has_body,
        src_locs,
        null,
    );
}

/// Given a library name, examines if the library name should end up in
/// `link.File.Options.system_libs` table (for example, libc is always
/// specified via dedicated flag `link.File.Options.link_libc` instead),
/// and puts it there if it doesn't exist.
/// It also dupes the library name which can then be saved as part of the
/// respective `Decl` (either `ExternFn` or `Var`).
/// The liveness of the duped library name is tied to liveness of `Module`.
/// To deallocate, call `deinit` on the respective `Decl` (`ExternFn` or `Var`).
fn handleExternLibName(
    sema: *Sema,
    block: *Block,
    src_loc: LazySrcLoc,
    lib_name: []const u8,
) CompileError![:0]u8 {
    blk: {
        const mod = sema.mod;
        const target = mod.getTarget();
        log.debug("extern fn symbol expected in lib '{s}'", .{lib_name});
        if (target_util.is_libc_lib_name(target, lib_name)) {
            if (!mod.comp.bin_file.options.link_libc) {
                return sema.fail(
                    block,
                    src_loc,
                    "dependency on libc must be explicitly specified in the build command",
                    .{},
                );
            }
            mod.comp.bin_file.options.link_libc = true;
            break :blk;
        }
        if (target_util.is_libcpp_lib_name(target, lib_name)) {
            if (!mod.comp.bin_file.options.link_libcpp) {
                return sema.fail(
                    block,
                    src_loc,
                    "dependency on libc++ must be explicitly specified in the build command",
                    .{},
                );
            }
            mod.comp.bin_file.options.link_libcpp = true;
            break :blk;
        }
        if (mem.eql(u8, lib_name, "unwind")) {
            mod.comp.bin_file.options.link_libunwind = true;
            break :blk;
        }
        if (!target.isWasm() and !mod.comp.bin_file.options.pic) {
            return sema.fail(
                block,
                src_loc,
                "dependency on dynamic library '{s}' requires enabling Position Independent Code. Fixed by `-l{s}` or `-fPIC`.",
                .{ lib_name, lib_name },
            );
        }
        mod.comp.stage1AddLinkLib(lib_name) catch |err| {
            return sema.fail(block, src_loc, "unable to add link lib '{s}': {s}", .{
                lib_name, @errorName(err),
            });
        };
    }
    return sema.gpa.dupeZ(u8, lib_name);
}

fn funcCommon(
    sema: *Sema,
    block: *Block,
    src_node_offset: i32,
    func_inst: Zir.Inst.Index,
    ret_ty_body: []const Zir.Inst.Index,
    cc: std.builtin.CallingConvention,
    align_val: Value,
    var_args: bool,
    inferred_error_set: bool,
    is_extern: bool,
    has_body: bool,
    src_locs: Zir.Inst.Func.SrcLocs,
    opt_lib_name: ?[]const u8,
) CompileError!Air.Inst.Ref {
    const ret_ty_src: LazySrcLoc = .{ .node_offset_fn_type_ret_ty = src_node_offset };

    // The return type body might be a type expression that depends on generic parameters.
    // In such case we need to use a generic_poison value for the return type and mark
    // the function as generic.
    var is_generic = false;
    const bare_return_type: Type = ret_ty: {
        if (ret_ty_body.len == 0) break :ret_ty Type.void;

        const err = err: {
            // Make sure any nested param instructions don't clobber our work.
            const prev_params = block.params;
            block.params = .{};
            defer {
                block.params.deinit(sema.gpa);
                block.params = prev_params;
            }
            if (sema.resolveBody(block, ret_ty_body, func_inst)) |ret_ty_inst| {
                if (sema.analyzeAsType(block, ret_ty_src, ret_ty_inst)) |ret_ty| {
                    break :ret_ty ret_ty;
                } else |err| break :err err;
            } else |err| break :err err;
            // Check for generic params.
            for (block.params.items) |param| {
                if (param.ty.tag() == .generic_poison) is_generic = true;
            }
        };
        switch (err) {
            error.GenericPoison => {
                // The type is not available until the generic instantiation.
                is_generic = true;
                break :ret_ty Type.initTag(.generic_poison);
            },
            else => |e| return e,
        }
    };

    const mod = sema.mod;

    const new_func: *Module.Fn = new_func: {
        if (!has_body) break :new_func undefined;
        if (sema.comptime_args_fn_inst == func_inst) {
            const new_func = sema.preallocated_new_func.?;
            sema.preallocated_new_func = null; // take ownership
            break :new_func new_func;
        }
        break :new_func try sema.gpa.create(Module.Fn);
    };
    errdefer if (has_body) sema.gpa.destroy(new_func);

    var maybe_inferred_error_set_node: ?*Module.Fn.InferredErrorSetListNode = null;
    errdefer if (maybe_inferred_error_set_node) |node| sema.gpa.destroy(node);
    // Note: no need to errdefer since this will still be in its default state at the end of the function.

    const target = mod.getTarget();

    const fn_ty: Type = fn_ty: {
        const alignment: u32 = if (align_val.tag() == .null_value) 0 else a: {
            const alignment = @intCast(u32, align_val.toUnsignedInt(target));
            if (alignment == target_util.defaultFunctionAlignment(target)) {
                break :a 0;
            } else {
                break :a alignment;
            }
        };

        // Hot path for some common function types.
        // TODO can we eliminate some of these Type tag values? seems unnecessarily complicated.
        if (!is_generic and block.params.items.len == 0 and !var_args and
            alignment == 0 and !inferred_error_set)
        {
            if (bare_return_type.zigTypeTag() == .NoReturn and cc == .Unspecified) {
                break :fn_ty Type.initTag(.fn_noreturn_no_args);
            }

            if (bare_return_type.zigTypeTag() == .Void and cc == .Unspecified) {
                break :fn_ty Type.initTag(.fn_void_no_args);
            }

            if (bare_return_type.zigTypeTag() == .NoReturn and cc == .Naked) {
                break :fn_ty Type.initTag(.fn_naked_noreturn_no_args);
            }

            if (bare_return_type.zigTypeTag() == .Void and cc == .C) {
                break :fn_ty Type.initTag(.fn_ccc_void_no_args);
            }
        }

        const param_types = try sema.arena.alloc(Type, block.params.items.len);
        const comptime_params = try sema.arena.alloc(bool, block.params.items.len);
        for (block.params.items) |param, i| {
            const param_src: LazySrcLoc = .{ .node_offset = src_node_offset }; // TODO better src
            param_types[i] = param.ty;
            comptime_params[i] = param.is_comptime or
                try sema.typeRequiresComptime(block, param_src, param.ty);
            is_generic = is_generic or comptime_params[i] or param.ty.tag() == .generic_poison;
            if (is_extern and is_generic) {
                // TODO add note: function is generic because of this parameter
                return sema.fail(block, param_src, "extern function cannot be generic", .{});
            }
        }

        const ret_poison = if (!is_generic) rp: {
            if (sema.typeRequiresComptime(block, ret_ty_src, bare_return_type)) |ret_comptime| {
                is_generic = ret_comptime;
                break :rp bare_return_type.tag() == .generic_poison;
            } else |err| switch (err) {
                error.GenericPoison => {
                    is_generic = true;
                    break :rp true;
                },
                else => |e| return e,
            }
        } else bare_return_type.tag() == .generic_poison;

        const return_type = if (!inferred_error_set or ret_poison)
            bare_return_type
        else blk: {
            const node = try sema.gpa.create(Module.Fn.InferredErrorSetListNode);
            node.data = .{ .func = new_func };
            maybe_inferred_error_set_node = node;

            const error_set_ty = try Type.Tag.error_set_inferred.create(sema.arena, &node.data);
            break :blk try Type.Tag.error_union.create(sema.arena, .{
                .error_set = error_set_ty,
                .payload = bare_return_type,
            });
        };

        break :fn_ty try Type.Tag.function.create(sema.arena, .{
            .param_types = param_types,
            .comptime_params = comptime_params.ptr,
            .return_type = return_type,
            .cc = cc,
            .alignment = alignment,
            .is_var_args = var_args,
            .is_generic = is_generic,
        });
    };

    if (is_extern) {
        const new_extern_fn = try sema.gpa.create(Module.ExternFn);
        errdefer sema.gpa.destroy(new_extern_fn);

        new_extern_fn.* = Module.ExternFn{
            .owner_decl = sema.owner_decl,
            .lib_name = null,
        };

        if (opt_lib_name) |lib_name| {
            new_extern_fn.lib_name = try sema.handleExternLibName(block, .{
                .node_offset_lib_name = src_node_offset,
            }, lib_name);
        }

        const extern_fn_payload = try sema.arena.create(Value.Payload.ExternFn);
        extern_fn_payload.* = .{
            .base = .{ .tag = .extern_fn },
            .data = new_extern_fn,
        };
        return sema.addConstant(fn_ty, Value.initPayload(&extern_fn_payload.base));
    }

    if (!has_body) {
        return sema.addType(fn_ty);
    }

    const is_inline = fn_ty.fnCallingConvention() == .Inline;
    const anal_state: Module.Fn.Analysis = if (is_inline) .inline_only else .queued;

    const comptime_args: ?[*]TypedValue = if (sema.comptime_args_fn_inst == func_inst) blk: {
        break :blk if (sema.comptime_args.len == 0) null else sema.comptime_args.ptr;
    } else null;

    const param_names = try sema.gpa.alloc([:0]const u8, block.params.items.len);
    for (param_names) |*param_name, i| {
        param_name.* = try sema.gpa.dupeZ(u8, block.params.items[i].name);
    }

    const hash = new_func.hash;
    const fn_payload = try sema.arena.create(Value.Payload.Function);
    new_func.* = .{
        .state = anal_state,
        .zir_body_inst = func_inst,
        .owner_decl = sema.owner_decl,
        .comptime_args = comptime_args,
        .hash = hash,
        .lbrace_line = src_locs.lbrace_line,
        .rbrace_line = src_locs.rbrace_line,
        .lbrace_column = @truncate(u16, src_locs.columns),
        .rbrace_column = @truncate(u16, src_locs.columns >> 16),
        .param_names = param_names,
    };
    if (maybe_inferred_error_set_node) |node| {
        new_func.inferred_error_sets.prepend(node);
    }
    maybe_inferred_error_set_node = null;
    fn_payload.* = .{
        .base = .{ .tag = .function },
        .data = new_func,
    };
    return sema.addConstant(fn_ty, Value.initPayload(&fn_payload.base));
}

fn zirParam(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    comptime_syntax: bool,
) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_tok;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.Param, inst_data.payload_index);
    const param_name = sema.code.nullTerminatedString(extra.data.name);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];

    // We could be in a generic function instantiation, or we could be evaluating a generic
    // function without any comptime args provided.
    const param_ty = param_ty: {
        const err = err: {
            // Make sure any nested param instructions don't clobber our work.
            const prev_params = block.params;
            const prev_preallocated_new_func = sema.preallocated_new_func;
            block.params = .{};
            sema.preallocated_new_func = null;
            defer {
                block.params.deinit(sema.gpa);
                block.params = prev_params;
                sema.preallocated_new_func = prev_preallocated_new_func;
            }

            if (sema.resolveBody(block, body, inst)) |param_ty_inst| {
                if (sema.analyzeAsType(block, src, param_ty_inst)) |param_ty| {
                    if (param_ty.zigTypeTag() == .Fn and param_ty.fnInfo().is_generic) {
                        // zirFunc will not emit error.GenericPoison to build a
                        // partial type for generic functions but we still need to
                        // detect if a function parameter is a generic function
                        // to force the parent function to also be generic.
                        break :err error.GenericPoison;
                    }
                    break :param_ty param_ty;
                } else |err| break :err err;
            } else |err| break :err err;
        };
        switch (err) {
            error.GenericPoison => {
                // The type is not available until the generic instantiation.
                // We result the param instruction with a poison value and
                // insert an anytype parameter.
                try block.params.append(sema.gpa, .{
                    .ty = Type.initTag(.generic_poison),
                    .is_comptime = comptime_syntax,
                    .name = param_name,
                });
                try sema.inst_map.putNoClobber(sema.gpa, inst, .generic_poison);
                return;
            },
            else => |e| return e,
        }
    };
    const is_comptime = comptime_syntax or
        try sema.typeRequiresComptime(block, src, param_ty);
    if (sema.inst_map.get(inst)) |arg| {
        if (is_comptime) {
            // We have a comptime value for this parameter so it should be elided from the
            // function type of the function instruction in this block.
            const coerced_arg = try sema.coerce(block, param_ty, arg, src);
            sema.inst_map.putAssumeCapacity(inst, coerced_arg);
            return;
        }
        // Even though a comptime argument is provided, the generic function wants to treat
        // this as a runtime parameter.
        assert(sema.inst_map.remove(inst));
    }

    if (sema.preallocated_new_func != null) {
        if (try sema.typeHasOnePossibleValue(block, src, param_ty)) |opv| {
            // In this case we are instantiating a generic function call with a non-comptime
            // non-anytype parameter that ended up being a one-possible-type.
            // We don't want the parameter to be part of the instantiated function type.
            const result = try sema.addConstant(param_ty, opv);
            try sema.inst_map.put(sema.gpa, inst, result);
            return;
        }
    }

    try block.params.append(sema.gpa, .{
        .ty = param_ty,
        .is_comptime = is_comptime,
        .name = param_name,
    });
    const result = try sema.addConstant(param_ty, Value.initTag(.generic_poison));
    try sema.inst_map.putNoClobber(sema.gpa, inst, result);
}

fn zirParamAnytype(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    comptime_syntax: bool,
) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const src = inst_data.src();
    const param_name = inst_data.get(sema.code);

    if (sema.inst_map.get(inst)) |air_ref| {
        const param_ty = sema.typeOf(air_ref);
        if (comptime_syntax or try sema.typeRequiresComptime(block, src, param_ty)) {
            // We have a comptime value for this parameter so it should be elided from the
            // function type of the function instruction in this block.
            return;
        }
        if (null != try sema.typeHasOnePossibleValue(block, src, param_ty)) {
            return;
        }
        // The map is already populated but we do need to add a runtime parameter.
        try block.params.append(sema.gpa, .{
            .ty = param_ty,
            .is_comptime = false,
            .name = param_name,
        });
        return;
    }

    // We are evaluating a generic function without any comptime args provided.

    try block.params.append(sema.gpa, .{
        .ty = Type.initTag(.generic_poison),
        .is_comptime = comptime_syntax,
        .name = param_name,
    });
    try sema.inst_map.put(sema.gpa, inst, .generic_poison);
}

fn zirAs(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    return sema.analyzeAs(block, sema.src, bin_inst.lhs, bin_inst.rhs);
}

fn zirAsNode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.As, inst_data.payload_index).data;
    return sema.analyzeAs(block, src, extra.dest_type, extra.operand);
}

fn analyzeAs(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_dest_type: Zir.Inst.Ref,
    zir_operand: Zir.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const dest_ty = try sema.resolveType(block, src, zir_dest_type);
    const operand = sema.resolveInst(zir_operand);
    if (dest_ty.tag() == .var_args_param) return operand;
    return sema.coerce(block, dest_ty, operand, src);
}

fn zirPtrToInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ptr_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr = sema.resolveInst(inst_data.operand);
    const ptr_ty = sema.typeOf(ptr);
    if (!ptr_ty.isPtrAtRuntime()) {
        const target = sema.mod.getTarget();
        return sema.fail(block, ptr_src, "expected pointer, found '{}'", .{ptr_ty.fmt(target)});
    }
    if (try sema.resolveMaybeUndefVal(block, ptr_src, ptr)) |ptr_val| {
        return sema.addConstant(Type.usize, ptr_val);
    }
    try sema.requireRuntimeBlock(block, ptr_src);
    return block.addUnOp(.ptrtoint, ptr);
}

fn zirFieldVal(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_field_name = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Field, inst_data.payload_index).data;
    const field_name = sema.code.nullTerminatedString(extra.field_name_start);
    const object = sema.resolveInst(extra.lhs);
    return sema.fieldVal(block, src, object, field_name, field_name_src);
}

fn zirFieldPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_field_name = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Field, inst_data.payload_index).data;
    const field_name = sema.code.nullTerminatedString(extra.field_name_start);
    const object_ptr = sema.resolveInst(extra.lhs);
    return sema.fieldPtr(block, src, object_ptr, field_name, field_name_src);
}

fn zirFieldCallBind(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_field_name = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Field, inst_data.payload_index).data;
    const field_name = sema.code.nullTerminatedString(extra.field_name_start);
    const object_ptr = sema.resolveInst(extra.lhs);
    return sema.fieldCallBind(block, src, object_ptr, field_name, field_name_src);
}

fn zirFieldValNamed(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.FieldNamed, inst_data.payload_index).data;
    const object = sema.resolveInst(extra.lhs);
    const field_name = try sema.resolveConstString(block, field_name_src, extra.field_name);
    return sema.fieldVal(block, src, object, field_name, field_name_src);
}

fn zirFieldPtrNamed(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.FieldNamed, inst_data.payload_index).data;
    const object_ptr = sema.resolveInst(extra.lhs);
    const field_name = try sema.resolveConstString(block, field_name_src, extra.field_name);
    return sema.fieldPtr(block, src, object_ptr, field_name, field_name_src);
}

fn zirFieldCallBindNamed(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const field_name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.FieldNamed, inst_data.payload_index).data;
    const object_ptr = sema.resolveInst(extra.lhs);
    const field_name = try sema.resolveConstString(block, field_name_src, extra.field_name);
    return sema.fieldCallBind(block, src, object_ptr, field_name, field_name_src);
}

fn zirIntCast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;

    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);

    return sema.intCast(block, dest_ty, dest_ty_src, operand, operand_src, true);
}

fn intCast(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    dest_ty_src: LazySrcLoc,
    operand: Air.Inst.Ref,
    operand_src: LazySrcLoc,
    runtime_safety: bool,
) CompileError!Air.Inst.Ref {
    const dest_is_comptime_int = try sema.checkIntType(block, dest_ty_src, dest_ty);
    _ = try sema.checkIntType(block, operand_src, sema.typeOf(operand));

    if (try sema.isComptimeKnown(block, operand_src, operand)) {
        return sema.coerce(block, dest_ty, operand, operand_src);
    } else if (dest_is_comptime_int) {
        return sema.fail(block, operand_src, "unable to cast runtime value to 'comptime_int'", .{});
    }

    if ((try sema.typeHasOnePossibleValue(block, dest_ty_src, dest_ty))) |opv| {
        // requirement: intCast(u0, input) iff input == 0
        if (runtime_safety and block.wantSafety()) {
            try sema.requireRuntimeBlock(block, operand_src);
            const target = sema.mod.getTarget();
            const wanted_info = dest_ty.intInfo(target);
            const wanted_bits = wanted_info.bits;

            if (wanted_bits == 0) {
                const zero_inst = try sema.addConstant(sema.typeOf(operand), Value.zero);
                const is_in_range = try block.addBinOp(.cmp_eq, operand, zero_inst);
                try sema.addSafetyCheck(block, is_in_range, .cast_truncated_data);
            }
        }

        return sema.addConstant(dest_ty, opv);
    }

    try sema.requireRuntimeBlock(block, operand_src);
    if (runtime_safety and block.wantSafety()) {
        const target = sema.mod.getTarget();
        const operand_ty = sema.typeOf(operand);
        const actual_info = operand_ty.intInfo(target);
        const wanted_info = dest_ty.intInfo(target);
        const actual_bits = actual_info.bits;
        const wanted_bits = wanted_info.bits;

        // requirement: signed to unsigned >= 0
        if (actual_info.signedness == .signed and
            wanted_info.signedness == .unsigned)
        {
            const zero_inst = try sema.addConstant(sema.typeOf(operand), Value.zero);
            const is_in_range = try block.addBinOp(.cmp_gte, operand, zero_inst);
            try sema.addSafetyCheck(block, is_in_range, .cast_truncated_data);
        }

        // requirement: unsigned int value fits into target type
        if (actual_bits > wanted_bits or
            (actual_bits == wanted_bits and
            actual_info.signedness == .unsigned and
            wanted_info.signedness == .signed))
        {
            const max_int = try dest_ty.maxInt(sema.arena, target);
            const max_int_inst = try sema.addConstant(operand_ty, max_int);
            const is_in_range = try block.addBinOp(.cmp_lte, operand, max_int_inst);
            try sema.addSafetyCheck(block, is_in_range, .cast_truncated_data);
        }
    }
    return block.addTyOp(.intcast, dest_ty, operand);
}

fn zirBitcast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const target = sema.mod.getTarget();

    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    switch (dest_ty.zigTypeTag()) {
        .AnyFrame,
        .ComptimeFloat,
        .ComptimeInt,
        .Enum,
        .EnumLiteral,
        .ErrorSet,
        .ErrorUnion,
        .Fn,
        .Frame,
        .NoReturn,
        .Null,
        .Opaque,
        .Optional,
        .Type,
        .Undefined,
        .Void,
        => return sema.fail(block, dest_ty_src, "invalid type '{}' for @bitCast", .{dest_ty.fmt(target)}),

        .Pointer => return sema.fail(block, dest_ty_src, "cannot @bitCast to '{}', use @ptrCast to cast to a pointer", .{
            dest_ty.fmt(target),
        }),
        .Struct, .Union => if (dest_ty.containerLayout() == .Auto) {
            const container = switch (dest_ty.zigTypeTag()) {
                .Struct => "struct",
                .Union => "union",
                else => unreachable,
            };
            return sema.fail(block, dest_ty_src, "cannot @bitCast to '{}', {s} does not have a guaranteed in-memory layout", .{
                dest_ty.fmt(target), container,
            });
        },
        .BoundFn => @panic("TODO remove this type from the language and compiler"),

        .Array,
        .Bool,
        .Float,
        .Int,
        .Vector,
        => {},
    }

    const operand = sema.resolveInst(extra.rhs);
    return sema.bitCast(block, dest_ty, operand, operand_src);
}

fn zirFloatCast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;

    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);

    const target = sema.mod.getTarget();
    const dest_is_comptime_float = switch (dest_ty.zigTypeTag()) {
        .ComptimeFloat => true,
        .Float => false,
        else => return sema.fail(
            block,
            dest_ty_src,
            "expected float type, found '{}'",
            .{dest_ty.fmt(target)},
        ),
    };

    const operand_ty = sema.typeOf(operand);
    switch (operand_ty.zigTypeTag()) {
        .ComptimeFloat, .Float, .ComptimeInt => {},
        else => return sema.fail(
            block,
            operand_src,
            "expected float type, found '{}'",
            .{operand_ty.fmt(target)},
        ),
    }

    if (try sema.isComptimeKnown(block, operand_src, operand)) {
        return sema.coerce(block, dest_ty, operand, operand_src);
    }
    if (dest_is_comptime_float) {
        return sema.fail(block, src, "unable to cast runtime value to 'comptime_float'", .{});
    }
    const src_bits = operand_ty.floatBits(target);
    const dst_bits = dest_ty.floatBits(target);
    if (dst_bits >= src_bits) {
        return sema.coerce(block, dest_ty, operand, operand_src);
    }
    try sema.requireRuntimeBlock(block, operand_src);
    return block.addTyOp(.fptrunc, dest_ty, operand);
}

fn zirElemVal(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const array = sema.resolveInst(bin_inst.lhs);
    const elem_index = sema.resolveInst(bin_inst.rhs);
    return sema.elemVal(block, sema.src, array, elem_index, sema.src);
}

fn zirElemValNode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const elem_index_src: LazySrcLoc = .{ .node_offset_array_access_index = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const array = sema.resolveInst(extra.lhs);
    const elem_index = sema.resolveInst(extra.rhs);
    return sema.elemVal(block, src, array, elem_index, elem_index_src);
}

fn zirElemPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const bin_inst = sema.code.instructions.items(.data)[inst].bin;
    const array_ptr = sema.resolveInst(bin_inst.lhs);
    const elem_index = sema.resolveInst(bin_inst.rhs);
    return sema.elemPtr(block, sema.src, array_ptr, elem_index, sema.src);
}

fn zirElemPtrNode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const elem_index_src: LazySrcLoc = .{ .node_offset_array_access_index = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const array_ptr = sema.resolveInst(extra.lhs);
    const elem_index = sema.resolveInst(extra.rhs);
    return sema.elemPtr(block, src, array_ptr, elem_index, elem_index_src);
}

fn zirElemPtrImm(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.ElemPtrImm, inst_data.payload_index).data;
    const array_ptr = sema.resolveInst(extra.ptr);
    const elem_index = try sema.addIntUnsigned(Type.usize, extra.index);
    return sema.elemPtr(block, src, array_ptr, elem_index, src);
}

fn zirSliceStart(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.SliceStart, inst_data.payload_index).data;
    const array_ptr = sema.resolveInst(extra.lhs);
    const start = sema.resolveInst(extra.start);

    return sema.analyzeSlice(block, src, array_ptr, start, .none, .none, .unneeded);
}

fn zirSliceEnd(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.SliceEnd, inst_data.payload_index).data;
    const array_ptr = sema.resolveInst(extra.lhs);
    const start = sema.resolveInst(extra.start);
    const end = sema.resolveInst(extra.end);

    return sema.analyzeSlice(block, src, array_ptr, start, end, .none, .unneeded);
}

fn zirSliceSentinel(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const sentinel_src: LazySrcLoc = .{ .node_offset_slice_sentinel = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.SliceSentinel, inst_data.payload_index).data;
    const array_ptr = sema.resolveInst(extra.lhs);
    const start = sema.resolveInst(extra.start);
    const end = sema.resolveInst(extra.end);
    const sentinel = sema.resolveInst(extra.sentinel);

    return sema.analyzeSlice(block, src, array_ptr, start, end, sentinel, sentinel_src);
}

fn zirSwitchCapture(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_multi: bool,
    is_ref: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const zir_datas = sema.code.instructions.items(.data);
    const capture_info = zir_datas[inst].switch_capture;
    const switch_info = zir_datas[capture_info.switch_inst].pl_node;
    const switch_extra = sema.code.extraData(Zir.Inst.SwitchBlock, switch_info.payload_index);
    const operand_src: LazySrcLoc = .{ .node_offset_switch_operand = switch_info.src_node };
    const switch_src = switch_info.src();
    const operand_is_ref = switch_extra.data.bits.is_ref;
    const cond_inst = Zir.refToIndex(switch_extra.data.operand).?;
    const cond_info = sema.code.instructions.items(.data)[cond_inst].un_node;
    const operand_ptr = sema.resolveInst(cond_info.operand);
    const operand_ptr_ty = sema.typeOf(operand_ptr);
    const operand_ty = if (operand_is_ref) operand_ptr_ty.childType() else operand_ptr_ty;
    const target = sema.mod.getTarget();

    const operand = if (operand_is_ref)
        try sema.analyzeLoad(block, operand_src, operand_ptr, operand_src)
    else
        operand_ptr;

    if (capture_info.prong_index == std.math.maxInt(@TypeOf(capture_info.prong_index))) {
        // It is the else/`_` prong.
        if (is_ref) {
            assert(operand_is_ref);
            return operand_ptr;
        }

        switch (operand_ty.zigTypeTag()) {
            .ErrorSet => return sema.bitCast(block, block.switch_else_err_ty.?, operand, operand_src),
            else => return operand,
        }
    }

    const items = if (is_multi)
        switch_extra.data.getMultiProng(sema.code, switch_extra.end, capture_info.prong_index).items
    else
        &[_]Zir.Inst.Ref{
            switch_extra.data.getScalarProng(sema.code, switch_extra.end, capture_info.prong_index).item,
        };

    switch (operand_ty.zigTypeTag()) {
        .Union => {
            const union_obj = operand_ty.cast(Type.Payload.Union).?.data;
            const enum_ty = union_obj.tag_ty;

            const first_item = sema.resolveInst(items[0]);
            // Previous switch validation ensured this will succeed
            const first_item_val = sema.resolveConstValue(block, .unneeded, first_item) catch unreachable;

            const first_field_index = @intCast(u32, enum_ty.enumTagFieldIndex(first_item_val, target).?);
            const first_field = union_obj.fields.values()[first_field_index];

            for (items[1..]) |item| {
                const item_ref = sema.resolveInst(item);
                // Previous switch validation ensured this will succeed
                const item_val = sema.resolveConstValue(block, .unneeded, item_ref) catch unreachable;

                const field_index = enum_ty.enumTagFieldIndex(item_val, target).?;
                const field = union_obj.fields.values()[field_index];
                if (!field.ty.eql(first_field.ty, target)) {
                    const first_item_src = switch_src; // TODO better source location
                    const item_src = switch_src;
                    const msg = msg: {
                        const msg = try sema.errMsg(block, switch_src, "capture group with incompatible types", .{});
                        errdefer msg.destroy(sema.gpa);
                        try sema.errNote(block, first_item_src, msg, "type '{}' here", .{first_field.ty.fmt(target)});
                        try sema.errNote(block, item_src, msg, "type '{}' here", .{field.ty.fmt(target)});
                        break :msg msg;
                    };
                    return sema.failWithOwnedErrorMsg(block, msg);
                }
            }

            if (is_ref) {
                assert(operand_is_ref);

                const field_ty_ptr = try Type.ptr(sema.arena, target, .{
                    .pointee_type = first_field.ty,
                    .@"addrspace" = .generic,
                    .mutable = operand_ptr_ty.ptrIsMutable(),
                });

                if (try sema.resolveDefinedValue(block, operand_src, operand_ptr)) |op_ptr_val| {
                    return sema.addConstant(
                        field_ty_ptr,
                        try Value.Tag.field_ptr.create(sema.arena, .{
                            .container_ptr = op_ptr_val,
                            .container_ty = operand_ty,
                            .field_index = first_field_index,
                        }),
                    );
                }
                try sema.requireRuntimeBlock(block, operand_src);
                return block.addStructFieldPtr(operand_ptr, first_field_index, field_ty_ptr);
            }

            if (try sema.resolveDefinedValue(block, operand_src, operand)) |operand_val| {
                return sema.addConstant(
                    first_field.ty,
                    operand_val.castTag(.@"union").?.data.val,
                );
            }
            try sema.requireRuntimeBlock(block, operand_src);
            return block.addStructFieldVal(operand, first_field_index, first_field.ty);
        },
        .ErrorSet => {
            if (is_multi) {
                var names: Module.ErrorSet.NameMap = .{};
                try names.ensureUnusedCapacity(sema.arena, items.len);
                for (items) |item| {
                    const item_ref = sema.resolveInst(item);
                    // Previous switch validation ensured this will succeed
                    const item_val = sema.resolveConstValue(block, .unneeded, item_ref) catch unreachable;
                    names.putAssumeCapacityNoClobber(
                        item_val.getError().?,
                        {},
                    );
                }
                // names must be sorted
                Module.ErrorSet.sortNames(&names);
                const else_error_ty = try Type.Tag.error_set_merged.create(sema.arena, names);

                return sema.bitCast(block, else_error_ty, operand, operand_src);
            } else {
                const item_ref = sema.resolveInst(items[0]);
                // Previous switch validation ensured this will succeed
                const item_val = sema.resolveConstValue(block, .unneeded, item_ref) catch unreachable;

                const item_ty = try Type.Tag.error_set_single.create(sema.arena, item_val.getError().?);
                return sema.bitCast(block, item_ty, operand, operand_src);
            }
        },
        else => {
            return sema.fail(block, operand_src, "switch on type '{}' provides no capture value", .{
                operand_ty.fmt(target),
            });
        },
    }
}

fn zirSwitchCond(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_ref: bool,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src = src; // TODO make this point at the switch operand
    const operand_ptr = sema.resolveInst(inst_data.operand);
    const operand = if (is_ref)
        try sema.analyzeLoad(block, src, operand_ptr, operand_src)
    else
        operand_ptr;
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    switch (operand_ty.zigTypeTag()) {
        .Type,
        .Void,
        .Bool,
        .Int,
        .Float,
        .ComptimeFloat,
        .ComptimeInt,
        .EnumLiteral,
        .Pointer,
        .Fn,
        .ErrorSet,
        .Enum,
        => {
            if ((try sema.typeHasOnePossibleValue(block, operand_src, operand_ty))) |opv| {
                return sema.addConstant(operand_ty, opv);
            }
            return operand;
        },

        .Union => {
            const union_ty = try sema.resolveTypeFields(block, operand_src, operand_ty);
            const enum_ty = union_ty.unionTagType() orelse {
                const msg = msg: {
                    const msg = try sema.errMsg(block, src, "switch on untagged union", .{});
                    errdefer msg.destroy(sema.gpa);
                    try sema.addDeclaredHereNote(msg, union_ty);
                    break :msg msg;
                };
                return sema.failWithOwnedErrorMsg(block, msg);
            };
            return sema.unionToTag(block, enum_ty, operand, src);
        },

        .ErrorUnion,
        .NoReturn,
        .Array,
        .Struct,
        .Undefined,
        .Null,
        .Optional,
        .BoundFn,
        .Opaque,
        .Vector,
        .Frame,
        .AnyFrame,
        => return sema.fail(block, src, "switch on type '{}'", .{operand_ty.fmt(target)}),
    }
}

const SwitchErrorSet = std.StringHashMap(Module.SwitchProngSrc);

fn zirSwitchBlock(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const gpa = sema.gpa;
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const src_node_offset = inst_data.src_node;
    const operand_src: LazySrcLoc = .{ .node_offset_switch_operand = src_node_offset };
    const special_prong_src: LazySrcLoc = .{ .node_offset_switch_special_prong = src_node_offset };
    const extra = sema.code.extraData(Zir.Inst.SwitchBlock, inst_data.payload_index);

    const operand = sema.resolveInst(extra.data.operand);

    var header_extra_index: usize = extra.end;

    const scalar_cases_len = extra.data.bits.scalar_cases_len;
    const multi_cases_len = if (extra.data.bits.has_multi_cases) blk: {
        const multi_cases_len = sema.code.extra[header_extra_index];
        header_extra_index += 1;
        break :blk multi_cases_len;
    } else 0;

    const special_prong = extra.data.bits.specialProng();
    const special: struct { body: []const Zir.Inst.Index, end: usize } = switch (special_prong) {
        .none => .{ .body = &.{}, .end = header_extra_index },
        .under, .@"else" => blk: {
            const body_len = sema.code.extra[header_extra_index];
            const extra_body_start = header_extra_index + 1;
            break :blk .{
                .body = sema.code.extra[extra_body_start..][0..body_len],
                .end = extra_body_start + body_len,
            };
        },
    };

    const operand_ty = sema.typeOf(operand);

    var else_error_ty: ?Type = null;

    // Validate usage of '_' prongs.
    if (special_prong == .under and !operand_ty.isNonexhaustiveEnum()) {
        const msg = msg: {
            const msg = try sema.errMsg(
                block,
                src,
                "'_' prong only allowed when switching on non-exhaustive enums",
                .{},
            );
            errdefer msg.destroy(gpa);
            try sema.errNote(
                block,
                special_prong_src,
                msg,
                "'_' prong here",
                .{},
            );
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const target = sema.mod.getTarget();

    // Validate for duplicate items, missing else prong, and invalid range.
    switch (operand_ty.zigTypeTag()) {
        .Enum => {
            var seen_fields = try gpa.alloc(?Module.SwitchProngSrc, operand_ty.enumFieldCount());
            defer gpa.free(seen_fields);

            mem.set(?Module.SwitchProngSrc, seen_fields, null);

            var extra_index: usize = special.end;
            {
                var scalar_i: u32 = 0;
                while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                    const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    extra_index += body_len;

                    try sema.validateSwitchItemEnum(
                        block,
                        seen_fields,
                        item_ref,
                        src_node_offset,
                        .{ .scalar = scalar_i },
                    );
                }
            }
            {
                var multi_i: u32 = 0;
                while (multi_i < multi_cases_len) : (multi_i += 1) {
                    const items_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const ranges_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const items = sema.code.refSlice(extra_index, items_len);
                    extra_index += items_len + body_len;

                    for (items) |item_ref, item_i| {
                        try sema.validateSwitchItemEnum(
                            block,
                            seen_fields,
                            item_ref,
                            src_node_offset,
                            .{ .multi = .{ .prong = multi_i, .item = @intCast(u32, item_i) } },
                        );
                    }

                    try sema.validateSwitchNoRange(block, ranges_len, operand_ty, src_node_offset);
                }
            }
            const all_tags_handled = for (seen_fields) |seen_src| {
                if (seen_src == null) break false;
            } else !operand_ty.isNonexhaustiveEnum();

            switch (special_prong) {
                .none => {
                    if (!all_tags_handled) {
                        const msg = msg: {
                            const msg = try sema.errMsg(
                                block,
                                src,
                                "switch must handle all possibilities",
                                .{},
                            );
                            errdefer msg.destroy(sema.gpa);
                            for (seen_fields) |seen_src, i| {
                                if (seen_src != null) continue;

                                const field_name = operand_ty.enumFieldName(i);

                                // TODO have this point to the tag decl instead of here
                                try sema.errNote(
                                    block,
                                    src,
                                    msg,
                                    "unhandled enumeration value: '{s}'",
                                    .{field_name},
                                );
                            }
                            try sema.mod.errNoteNonLazy(
                                operand_ty.declSrcLoc(),
                                msg,
                                "enum '{}' declared here",
                                .{operand_ty.fmt(target)},
                            );
                            break :msg msg;
                        };
                        return sema.failWithOwnedErrorMsg(block, msg);
                    }
                },
                .under => {
                    if (all_tags_handled) return sema.fail(
                        block,
                        special_prong_src,
                        "unreachable '_' prong; all cases already handled",
                        .{},
                    );
                },
                .@"else" => {
                    if (all_tags_handled) return sema.fail(
                        block,
                        special_prong_src,
                        "unreachable else prong; all cases already handled",
                        .{},
                    );
                },
            }
        },
        .ErrorSet => {
            var seen_errors = SwitchErrorSet.init(gpa);
            defer seen_errors.deinit();

            var extra_index: usize = special.end;
            {
                var scalar_i: u32 = 0;
                while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                    const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    extra_index += body_len;

                    try sema.validateSwitchItemError(
                        block,
                        &seen_errors,
                        item_ref,
                        src_node_offset,
                        .{ .scalar = scalar_i },
                    );
                }
            }
            {
                var multi_i: u32 = 0;
                while (multi_i < multi_cases_len) : (multi_i += 1) {
                    const items_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const ranges_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const items = sema.code.refSlice(extra_index, items_len);
                    extra_index += items_len + body_len;

                    for (items) |item_ref, item_i| {
                        try sema.validateSwitchItemError(
                            block,
                            &seen_errors,
                            item_ref,
                            src_node_offset,
                            .{ .multi = .{ .prong = multi_i, .item = @intCast(u32, item_i) } },
                        );
                    }

                    try sema.validateSwitchNoRange(block, ranges_len, operand_ty, src_node_offset);
                }
            }

            try sema.resolveInferredErrorSetTy(block, src, operand_ty);

            if (operand_ty.isAnyError()) {
                if (special_prong != .@"else") {
                    return sema.fail(
                        block,
                        src,
                        "switch must handle all possibilities",
                        .{},
                    );
                }
                else_error_ty = Type.@"anyerror";
            } else {
                var maybe_msg: ?*Module.ErrorMsg = null;
                errdefer if (maybe_msg) |msg| msg.destroy(sema.gpa);

                for (operand_ty.errorSetNames()) |error_name| {
                    if (!seen_errors.contains(error_name) and special_prong != .@"else") {
                        const msg = maybe_msg orelse blk: {
                            maybe_msg = try sema.errMsg(
                                block,
                                src,
                                "switch must handle all possibilities",
                                .{},
                            );
                            break :blk maybe_msg.?;
                        };

                        try sema.errNote(
                            block,
                            src,
                            msg,
                            "unhandled error value: error.{s}",
                            .{error_name},
                        );
                    }
                }

                if (maybe_msg) |msg| {
                    try sema.mod.errNoteNonLazy(
                        operand_ty.declSrcLoc(),
                        msg,
                        "error set '{}' declared here",
                        .{operand_ty.fmt(target)},
                    );
                    return sema.failWithOwnedErrorMsg(block, msg);
                }

                if (special_prong == .@"else" and seen_errors.count() == operand_ty.errorSetNames().len) {
                    return sema.fail(
                        block,
                        special_prong_src,
                        "unreachable else prong; all cases already handled",
                        .{},
                    );
                }

                const error_names = operand_ty.errorSetNames();
                var names: Module.ErrorSet.NameMap = .{};
                try names.ensureUnusedCapacity(sema.arena, error_names.len);
                for (error_names) |error_name| {
                    if (seen_errors.contains(error_name)) continue;

                    names.putAssumeCapacityNoClobber(error_name, {});
                }

                // names must be sorted
                Module.ErrorSet.sortNames(&names);
                else_error_ty = try Type.Tag.error_set_merged.create(sema.arena, names);
            }
        },
        .Union => return sema.fail(block, src, "TODO validate switch .Union", .{}),
        .Int, .ComptimeInt => {
            var range_set = RangeSet.init(gpa, target);
            defer range_set.deinit();

            var extra_index: usize = special.end;
            {
                var scalar_i: u32 = 0;
                while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                    const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    extra_index += body_len;

                    try sema.validateSwitchItem(
                        block,
                        &range_set,
                        item_ref,
                        operand_ty,
                        src_node_offset,
                        .{ .scalar = scalar_i },
                    );
                }
            }
            {
                var multi_i: u32 = 0;
                while (multi_i < multi_cases_len) : (multi_i += 1) {
                    const items_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const ranges_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const items = sema.code.refSlice(extra_index, items_len);
                    extra_index += items_len;

                    for (items) |item_ref, item_i| {
                        try sema.validateSwitchItem(
                            block,
                            &range_set,
                            item_ref,
                            operand_ty,
                            src_node_offset,
                            .{ .multi = .{ .prong = multi_i, .item = @intCast(u32, item_i) } },
                        );
                    }

                    var range_i: u32 = 0;
                    while (range_i < ranges_len) : (range_i += 1) {
                        const item_first = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                        extra_index += 1;
                        const item_last = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                        extra_index += 1;

                        try sema.validateSwitchRange(
                            block,
                            &range_set,
                            item_first,
                            item_last,
                            operand_ty,
                            src_node_offset,
                            .{ .range = .{ .prong = multi_i, .item = range_i } },
                        );
                    }

                    extra_index += body_len;
                }
            }

            check_range: {
                if (operand_ty.zigTypeTag() == .Int) {
                    var arena = std.heap.ArenaAllocator.init(gpa);
                    defer arena.deinit();

                    const min_int = try operand_ty.minInt(arena.allocator(), target);
                    const max_int = try operand_ty.maxInt(arena.allocator(), target);
                    if (try range_set.spans(min_int, max_int, operand_ty)) {
                        if (special_prong == .@"else") {
                            return sema.fail(
                                block,
                                special_prong_src,
                                "unreachable else prong; all cases already handled",
                                .{},
                            );
                        }
                        break :check_range;
                    }
                }
                if (special_prong != .@"else") {
                    return sema.fail(
                        block,
                        src,
                        "switch must handle all possibilities",
                        .{},
                    );
                }
            }
        },
        .Bool => {
            var true_count: u8 = 0;
            var false_count: u8 = 0;

            var extra_index: usize = special.end;
            {
                var scalar_i: u32 = 0;
                while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                    const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    extra_index += body_len;

                    try sema.validateSwitchItemBool(
                        block,
                        &true_count,
                        &false_count,
                        item_ref,
                        src_node_offset,
                        .{ .scalar = scalar_i },
                    );
                }
            }
            {
                var multi_i: u32 = 0;
                while (multi_i < multi_cases_len) : (multi_i += 1) {
                    const items_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const ranges_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const items = sema.code.refSlice(extra_index, items_len);
                    extra_index += items_len + body_len;

                    for (items) |item_ref, item_i| {
                        try sema.validateSwitchItemBool(
                            block,
                            &true_count,
                            &false_count,
                            item_ref,
                            src_node_offset,
                            .{ .multi = .{ .prong = multi_i, .item = @intCast(u32, item_i) } },
                        );
                    }

                    try sema.validateSwitchNoRange(block, ranges_len, operand_ty, src_node_offset);
                }
            }
            switch (special_prong) {
                .@"else" => {
                    if (true_count + false_count == 2) {
                        return sema.fail(
                            block,
                            src,
                            "unreachable else prong; all cases already handled",
                            .{},
                        );
                    }
                },
                .under, .none => {
                    if (true_count + false_count < 2) {
                        return sema.fail(
                            block,
                            src,
                            "switch must handle all possibilities",
                            .{},
                        );
                    }
                },
            }
        },
        .EnumLiteral, .Void, .Fn, .Pointer, .Type => {
            if (special_prong != .@"else") {
                return sema.fail(
                    block,
                    src,
                    "else prong required when switching on type '{}'",
                    .{operand_ty.fmt(target)},
                );
            }

            var seen_values = ValueSrcMap.initContext(gpa, .{
                .ty = operand_ty,
                .target = target,
            });
            defer seen_values.deinit();

            var extra_index: usize = special.end;
            {
                var scalar_i: u32 = 0;
                while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                    const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    extra_index += body_len;

                    try sema.validateSwitchItemSparse(
                        block,
                        &seen_values,
                        item_ref,
                        src_node_offset,
                        .{ .scalar = scalar_i },
                    );
                }
            }
            {
                var multi_i: u32 = 0;
                while (multi_i < multi_cases_len) : (multi_i += 1) {
                    const items_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const ranges_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const body_len = sema.code.extra[extra_index];
                    extra_index += 1;
                    const items = sema.code.refSlice(extra_index, items_len);
                    extra_index += items_len + body_len;

                    for (items) |item_ref, item_i| {
                        try sema.validateSwitchItemSparse(
                            block,
                            &seen_values,
                            item_ref,
                            src_node_offset,
                            .{ .multi = .{ .prong = multi_i, .item = @intCast(u32, item_i) } },
                        );
                    }

                    try sema.validateSwitchNoRange(block, ranges_len, operand_ty, src_node_offset);
                }
            }
        },

        .ErrorUnion,
        .NoReturn,
        .Array,
        .Struct,
        .Undefined,
        .Null,
        .Optional,
        .BoundFn,
        .Opaque,
        .Vector,
        .Frame,
        .AnyFrame,
        .ComptimeFloat,
        .Float,
        => return sema.fail(block, operand_src, "invalid switch operand type '{}'", .{
            operand_ty.fmt(target),
        }),
    }

    const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
    try sema.air_instructions.append(gpa, .{
        .tag = .block,
        .data = undefined,
    });
    var label: Block.Label = .{
        .zir_block = inst,
        .merges = .{
            .results = .{},
            .br_list = .{},
            .block_inst = block_inst,
        },
    };

    var child_block: Block = .{
        .parent = block,
        .sema = sema,
        .src_decl = block.src_decl,
        .namespace = block.namespace,
        .wip_capture_scope = block.wip_capture_scope,
        .instructions = .{},
        .label = &label,
        .inlining = block.inlining,
        .is_comptime = block.is_comptime,
        .switch_else_err_ty = else_error_ty,
    };
    const merges = &child_block.label.?.merges;
    defer child_block.instructions.deinit(gpa);
    defer merges.results.deinit(gpa);
    defer merges.br_list.deinit(gpa);

    if (try sema.resolveDefinedValue(&child_block, src, operand)) |operand_val| {
        var extra_index: usize = special.end;
        {
            var scalar_i: usize = 0;
            while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
                const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                extra_index += 1;
                const body_len = sema.code.extra[extra_index];
                extra_index += 1;
                const body = sema.code.extra[extra_index..][0..body_len];
                extra_index += body_len;

                const item = sema.resolveInst(item_ref);
                // Validation above ensured these will succeed.
                const item_val = sema.resolveConstValue(&child_block, .unneeded, item) catch unreachable;
                if (operand_val.eql(item_val, operand_ty, target)) {
                    return sema.resolveBlockBody(block, src, &child_block, body, inst, merges);
                }
            }
        }
        {
            var multi_i: usize = 0;
            while (multi_i < multi_cases_len) : (multi_i += 1) {
                const items_len = sema.code.extra[extra_index];
                extra_index += 1;
                const ranges_len = sema.code.extra[extra_index];
                extra_index += 1;
                const body_len = sema.code.extra[extra_index];
                extra_index += 1;
                const items = sema.code.refSlice(extra_index, items_len);
                extra_index += items_len;
                const body = sema.code.extra[extra_index + 2 * ranges_len ..][0..body_len];

                for (items) |item_ref| {
                    const item = sema.resolveInst(item_ref);
                    // Validation above ensured these will succeed.
                    const item_val = sema.resolveConstValue(&child_block, .unneeded, item) catch unreachable;
                    if (operand_val.eql(item_val, operand_ty, target)) {
                        return sema.resolveBlockBody(block, src, &child_block, body, inst, merges);
                    }
                }

                var range_i: usize = 0;
                while (range_i < ranges_len) : (range_i += 1) {
                    const item_first = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;
                    const item_last = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                    extra_index += 1;

                    // Validation above ensured these will succeed.
                    const first_tv = sema.resolveInstConst(&child_block, .unneeded, item_first) catch unreachable;
                    const last_tv = sema.resolveInstConst(&child_block, .unneeded, item_last) catch unreachable;
                    if (Value.compare(operand_val, .gte, first_tv.val, operand_ty, target) and
                        Value.compare(operand_val, .lte, last_tv.val, operand_ty, target))
                    {
                        return sema.resolveBlockBody(block, src, &child_block, body, inst, merges);
                    }
                }

                extra_index += body_len;
            }
        }
        return sema.resolveBlockBody(block, src, &child_block, special.body, inst, merges);
    }

    if (scalar_cases_len + multi_cases_len == 0) {
        if (special_prong == .none) {
            return sema.fail(block, src, "switch must handle all possibilities", .{});
        }
        return sema.resolveBlockBody(block, src, &child_block, special.body, inst, merges);
    }

    try sema.requireRuntimeBlock(block, src);

    const estimated_cases_extra = (scalar_cases_len + multi_cases_len) *
        @typeInfo(Air.SwitchBr.Case).Struct.fields.len + 2;
    var cases_extra = try std.ArrayListUnmanaged(u32).initCapacity(gpa, estimated_cases_extra);
    defer cases_extra.deinit(gpa);

    var case_block = child_block.makeSubBlock();
    case_block.runtime_loop = null;
    case_block.runtime_cond = operand_src;
    case_block.runtime_index += 1;
    defer case_block.instructions.deinit(gpa);

    var extra_index: usize = special.end;

    var scalar_i: usize = 0;
    while (scalar_i < scalar_cases_len) : (scalar_i += 1) {
        const item_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        const body_len = sema.code.extra[extra_index];
        extra_index += 1;
        const body = sema.code.extra[extra_index..][0..body_len];
        extra_index += body_len;

        var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, child_block.wip_capture_scope);
        defer wip_captures.deinit();

        case_block.instructions.shrinkRetainingCapacity(0);
        case_block.wip_capture_scope = wip_captures.scope;

        const item = sema.resolveInst(item_ref);
        // `item` is already guaranteed to be constant known.

        try sema.analyzeBody(&case_block, body);

        try wip_captures.finalize();

        try cases_extra.ensureUnusedCapacity(gpa, 3 + case_block.instructions.items.len);
        cases_extra.appendAssumeCapacity(1); // items_len
        cases_extra.appendAssumeCapacity(@intCast(u32, case_block.instructions.items.len));
        cases_extra.appendAssumeCapacity(@enumToInt(item));
        cases_extra.appendSliceAssumeCapacity(case_block.instructions.items);
    }

    var is_first = true;
    var prev_cond_br: Air.Inst.Index = undefined;
    var first_else_body: []const Air.Inst.Index = &.{};
    defer gpa.free(first_else_body);
    var prev_then_body: []const Air.Inst.Index = &.{};
    defer gpa.free(prev_then_body);

    var cases_len = scalar_cases_len;
    var multi_i: usize = 0;
    while (multi_i < multi_cases_len) : (multi_i += 1) {
        const items_len = sema.code.extra[extra_index];
        extra_index += 1;
        const ranges_len = sema.code.extra[extra_index];
        extra_index += 1;
        const body_len = sema.code.extra[extra_index];
        extra_index += 1;
        const items = sema.code.refSlice(extra_index, items_len);
        extra_index += items_len;

        case_block.instructions.shrinkRetainingCapacity(0);
        case_block.wip_capture_scope = child_block.wip_capture_scope;

        var any_ok: Air.Inst.Ref = .none;

        // If there are any ranges, we have to put all the items into the
        // else prong. Otherwise, we can take advantage of multiple items
        // mapping to the same body.
        if (ranges_len == 0) {
            cases_len += 1;

            const body = sema.code.extra[extra_index..][0..body_len];
            extra_index += body_len;
            try sema.analyzeBody(&case_block, body);

            try cases_extra.ensureUnusedCapacity(gpa, 2 + items.len +
                case_block.instructions.items.len);

            cases_extra.appendAssumeCapacity(@intCast(u32, items.len));
            cases_extra.appendAssumeCapacity(@intCast(u32, case_block.instructions.items.len));

            for (items) |item_ref| {
                const item = sema.resolveInst(item_ref);
                cases_extra.appendAssumeCapacity(@enumToInt(item));
            }

            cases_extra.appendSliceAssumeCapacity(case_block.instructions.items);
        } else {
            for (items) |item_ref| {
                const item = sema.resolveInst(item_ref);
                const cmp_ok = try case_block.addBinOp(.cmp_eq, operand, item);
                if (any_ok != .none) {
                    any_ok = try case_block.addBinOp(.bool_or, any_ok, cmp_ok);
                } else {
                    any_ok = cmp_ok;
                }
            }

            var range_i: usize = 0;
            while (range_i < ranges_len) : (range_i += 1) {
                const first_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                extra_index += 1;
                const last_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
                extra_index += 1;

                const item_first = sema.resolveInst(first_ref);
                const item_last = sema.resolveInst(last_ref);

                // operand >= first and operand <= last
                const range_first_ok = try case_block.addBinOp(
                    .cmp_gte,
                    operand,
                    item_first,
                );
                const range_last_ok = try case_block.addBinOp(
                    .cmp_lte,
                    operand,
                    item_last,
                );
                const range_ok = try case_block.addBinOp(
                    .bool_and,
                    range_first_ok,
                    range_last_ok,
                );
                if (any_ok != .none) {
                    any_ok = try case_block.addBinOp(.bool_or, any_ok, range_ok);
                } else {
                    any_ok = range_ok;
                }
            }

            const new_cond_br = try case_block.addInstAsIndex(.{ .tag = .cond_br, .data = .{
                .pl_op = .{
                    .operand = any_ok,
                    .payload = undefined,
                },
            } });
            var cond_body = case_block.instructions.toOwnedSlice(gpa);
            defer gpa.free(cond_body);

            var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, child_block.wip_capture_scope);
            defer wip_captures.deinit();

            case_block.instructions.shrinkRetainingCapacity(0);
            case_block.wip_capture_scope = wip_captures.scope;

            const body = sema.code.extra[extra_index..][0..body_len];
            extra_index += body_len;
            try sema.analyzeBody(&case_block, body);

            try wip_captures.finalize();

            if (is_first) {
                is_first = false;
                first_else_body = cond_body;
                cond_body = &.{};
            } else {
                try sema.air_extra.ensureUnusedCapacity(
                    gpa,
                    @typeInfo(Air.CondBr).Struct.fields.len + prev_then_body.len + cond_body.len,
                );

                sema.air_instructions.items(.data)[prev_cond_br].pl_op.payload =
                    sema.addExtraAssumeCapacity(Air.CondBr{
                    .then_body_len = @intCast(u32, prev_then_body.len),
                    .else_body_len = @intCast(u32, cond_body.len),
                });
                sema.air_extra.appendSliceAssumeCapacity(prev_then_body);
                sema.air_extra.appendSliceAssumeCapacity(cond_body);
            }
            gpa.free(prev_then_body);
            prev_then_body = case_block.instructions.toOwnedSlice(gpa);
            prev_cond_br = new_cond_br;
        }
    }

    var final_else_body: []const Air.Inst.Index = &.{};
    if (special.body.len != 0 or !is_first) {
        var wip_captures = try WipCaptureScope.init(gpa, sema.perm_arena, child_block.wip_capture_scope);
        defer wip_captures.deinit();

        case_block.instructions.shrinkRetainingCapacity(0);
        case_block.wip_capture_scope = wip_captures.scope;

        if (special.body.len != 0) {
            try sema.analyzeBody(&case_block, special.body);
        } else {
            // We still need a terminator in this block, but we have proven
            // that it is unreachable.
            // TODO this should be a special safety panic other than unreachable, something
            // like "panic: switch operand had corrupt value not allowed by the type"
            try case_block.addUnreachable(src, true);
        }

        try wip_captures.finalize();

        if (is_first) {
            final_else_body = case_block.instructions.items;
        } else {
            try sema.air_extra.ensureUnusedCapacity(gpa, prev_then_body.len +
                @typeInfo(Air.CondBr).Struct.fields.len + case_block.instructions.items.len);

            sema.air_instructions.items(.data)[prev_cond_br].pl_op.payload =
                sema.addExtraAssumeCapacity(Air.CondBr{
                .then_body_len = @intCast(u32, prev_then_body.len),
                .else_body_len = @intCast(u32, case_block.instructions.items.len),
            });
            sema.air_extra.appendSliceAssumeCapacity(prev_then_body);
            sema.air_extra.appendSliceAssumeCapacity(case_block.instructions.items);
            final_else_body = first_else_body;
        }
    }

    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.SwitchBr).Struct.fields.len +
        cases_extra.items.len + final_else_body.len);

    _ = try child_block.addInst(.{ .tag = .switch_br, .data = .{ .pl_op = .{
        .operand = operand,
        .payload = sema.addExtraAssumeCapacity(Air.SwitchBr{
            .cases_len = @intCast(u32, cases_len),
            .else_body_len = @intCast(u32, final_else_body.len),
        }),
    } } });
    sema.air_extra.appendSliceAssumeCapacity(cases_extra.items);
    sema.air_extra.appendSliceAssumeCapacity(final_else_body);

    return sema.analyzeBlockBody(block, src, &child_block, merges);
}

fn resolveSwitchItemVal(
    sema: *Sema,
    block: *Block,
    item_ref: Zir.Inst.Ref,
    switch_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
    range_expand: Module.SwitchProngSrc.RangeExpand,
) CompileError!TypedValue {
    const item = sema.resolveInst(item_ref);
    const item_ty = sema.typeOf(item);
    // Constructing a LazySrcLoc is costly because we only have the switch AST node.
    // Only if we know for sure we need to report a compile error do we resolve the
    // full source locations.
    if (sema.resolveConstValue(block, .unneeded, item)) |val| {
        return TypedValue{ .ty = item_ty, .val = val };
    } else |err| switch (err) {
        error.NeededSourceLocation => {
            const src = switch_prong_src.resolve(sema.gpa, block.src_decl, switch_node_offset, range_expand);
            return TypedValue{
                .ty = item_ty,
                .val = try sema.resolveConstValue(block, src, item),
            };
        },
        else => |e| return e,
    }
}

fn validateSwitchRange(
    sema: *Sema,
    block: *Block,
    range_set: *RangeSet,
    first_ref: Zir.Inst.Ref,
    last_ref: Zir.Inst.Ref,
    operand_ty: Type,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const first_val = (try sema.resolveSwitchItemVal(block, first_ref, src_node_offset, switch_prong_src, .first)).val;
    const last_val = (try sema.resolveSwitchItemVal(block, last_ref, src_node_offset, switch_prong_src, .last)).val;
    const maybe_prev_src = try range_set.add(first_val, last_val, operand_ty, switch_prong_src);
    return sema.validateSwitchDupe(block, maybe_prev_src, switch_prong_src, src_node_offset);
}

fn validateSwitchItem(
    sema: *Sema,
    block: *Block,
    range_set: *RangeSet,
    item_ref: Zir.Inst.Ref,
    operand_ty: Type,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const item_val = (try sema.resolveSwitchItemVal(block, item_ref, src_node_offset, switch_prong_src, .none)).val;
    const maybe_prev_src = try range_set.add(item_val, item_val, operand_ty, switch_prong_src);
    return sema.validateSwitchDupe(block, maybe_prev_src, switch_prong_src, src_node_offset);
}

fn validateSwitchItemEnum(
    sema: *Sema,
    block: *Block,
    seen_fields: []?Module.SwitchProngSrc,
    item_ref: Zir.Inst.Ref,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const item_tv = try sema.resolveSwitchItemVal(block, item_ref, src_node_offset, switch_prong_src, .none);
    const target = sema.mod.getTarget();
    const field_index = item_tv.ty.enumTagFieldIndex(item_tv.val, target) orelse {
        const msg = msg: {
            const src = switch_prong_src.resolve(sema.gpa, block.src_decl, src_node_offset, .none);
            const msg = try sema.errMsg(
                block,
                src,
                "enum '{}' has no tag with value '{}'",
                .{ item_tv.ty.fmt(target), item_tv.val.fmtValue(item_tv.ty, target) },
            );
            errdefer msg.destroy(sema.gpa);
            try sema.mod.errNoteNonLazy(
                item_tv.ty.declSrcLoc(),
                msg,
                "enum declared here",
                .{},
            );
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    };
    const maybe_prev_src = seen_fields[field_index];
    seen_fields[field_index] = switch_prong_src;
    return sema.validateSwitchDupe(block, maybe_prev_src, switch_prong_src, src_node_offset);
}

fn validateSwitchItemError(
    sema: *Sema,
    block: *Block,
    seen_errors: *SwitchErrorSet,
    item_ref: Zir.Inst.Ref,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const item_tv = try sema.resolveSwitchItemVal(block, item_ref, src_node_offset, switch_prong_src, .none);
    // TODO: Do i need to typecheck here?
    const error_name = item_tv.val.castTag(.@"error").?.data.name;
    const maybe_prev_src = if (try seen_errors.fetchPut(error_name, switch_prong_src)) |prev|
        prev.value
    else
        null;
    return sema.validateSwitchDupe(block, maybe_prev_src, switch_prong_src, src_node_offset);
}

fn validateSwitchDupe(
    sema: *Sema,
    block: *Block,
    maybe_prev_src: ?Module.SwitchProngSrc,
    switch_prong_src: Module.SwitchProngSrc,
    src_node_offset: i32,
) CompileError!void {
    const prev_prong_src = maybe_prev_src orelse return;
    const gpa = sema.gpa;
    const src = switch_prong_src.resolve(gpa, block.src_decl, src_node_offset, .none);
    const prev_src = prev_prong_src.resolve(gpa, block.src_decl, src_node_offset, .none);
    const msg = msg: {
        const msg = try sema.errMsg(
            block,
            src,
            "duplicate switch value",
            .{},
        );
        errdefer msg.destroy(sema.gpa);
        try sema.errNote(
            block,
            prev_src,
            msg,
            "previous value here",
            .{},
        );
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn validateSwitchItemBool(
    sema: *Sema,
    block: *Block,
    true_count: *u8,
    false_count: *u8,
    item_ref: Zir.Inst.Ref,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const item_val = (try sema.resolveSwitchItemVal(block, item_ref, src_node_offset, switch_prong_src, .none)).val;
    if (item_val.toBool()) {
        true_count.* += 1;
    } else {
        false_count.* += 1;
    }
    if (true_count.* + false_count.* > 2) {
        const src = switch_prong_src.resolve(sema.gpa, block.src_decl, src_node_offset, .none);
        return sema.fail(block, src, "duplicate switch value", .{});
    }
}

const ValueSrcMap = std.HashMap(Value, Module.SwitchProngSrc, Value.HashContext, std.hash_map.default_max_load_percentage);

fn validateSwitchItemSparse(
    sema: *Sema,
    block: *Block,
    seen_values: *ValueSrcMap,
    item_ref: Zir.Inst.Ref,
    src_node_offset: i32,
    switch_prong_src: Module.SwitchProngSrc,
) CompileError!void {
    const item_val = (try sema.resolveSwitchItemVal(block, item_ref, src_node_offset, switch_prong_src, .none)).val;
    const kv = (try seen_values.fetchPut(item_val, switch_prong_src)) orelse return;
    return sema.validateSwitchDupe(block, kv.value, switch_prong_src, src_node_offset);
}

fn validateSwitchNoRange(
    sema: *Sema,
    block: *Block,
    ranges_len: u32,
    operand_ty: Type,
    src_node_offset: i32,
) CompileError!void {
    if (ranges_len == 0)
        return;

    const operand_src: LazySrcLoc = .{ .node_offset_switch_operand = src_node_offset };
    const range_src: LazySrcLoc = .{ .node_offset_switch_range = src_node_offset };

    const target = sema.mod.getTarget();
    const msg = msg: {
        const msg = try sema.errMsg(
            block,
            operand_src,
            "ranges not allowed when switching on type '{}'",
            .{operand_ty.fmt(target)},
        );
        errdefer msg.destroy(sema.gpa);
        try sema.errNote(
            block,
            range_src,
            msg,
            "range here",
            .{},
        );
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn zirHasField(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const unresolved_ty = try sema.resolveType(block, ty_src, extra.lhs);
    const field_name = try sema.resolveConstString(block, name_src, extra.rhs);
    const ty = try sema.resolveTypeFields(block, ty_src, unresolved_ty);
    const target = sema.mod.getTarget();

    const has_field = hf: {
        if (ty.isSlice()) {
            if (mem.eql(u8, field_name, "ptr")) break :hf true;
            if (mem.eql(u8, field_name, "len")) break :hf true;
            break :hf false;
        }
        if (ty.castTag(.anon_struct)) |pl| {
            break :hf for (pl.data.names) |name| {
                if (mem.eql(u8, name, field_name)) break true;
            } else false;
        }
        if (ty.isTuple()) {
            const field_index = std.fmt.parseUnsigned(u32, field_name, 10) catch break :hf false;
            break :hf field_index < ty.structFieldCount();
        }
        break :hf switch (ty.zigTypeTag()) {
            .Struct => ty.structFields().contains(field_name),
            .Union => ty.unionFields().contains(field_name),
            .Enum => ty.enumFields().contains(field_name),
            .Array => mem.eql(u8, field_name, "len"),
            else => return sema.fail(block, ty_src, "type '{}' does not support '@hasField'", .{
                ty.fmt(target),
            }),
        };
    };
    if (has_field) {
        return Air.Inst.Ref.bool_true;
    } else {
        return Air.Inst.Ref.bool_false;
    }
}

fn zirHasDecl(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const container_type = try sema.resolveType(block, lhs_src, extra.lhs);
    const decl_name = try sema.resolveConstString(block, rhs_src, extra.rhs);

    try checkNamespaceType(sema, block, lhs_src, container_type);

    const namespace = container_type.getNamespace() orelse return Air.Inst.Ref.bool_false;
    if (try sema.lookupInNamespace(block, src, namespace, decl_name, true)) |decl| {
        if (decl.is_pub or decl.getFileScope() == block.getFileScope()) {
            return Air.Inst.Ref.bool_true;
        }
    }
    return Air.Inst.Ref.bool_false;
}

fn zirImport(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const mod = sema.mod;
    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const operand_src = inst_data.src();
    const operand = inst_data.get(sema.code);

    const result = mod.importFile(block.getFileScope(), operand) catch |err| switch (err) {
        error.ImportOutsidePkgPath => {
            return sema.fail(block, operand_src, "import of file outside package path: '{s}'", .{operand});
        },
        else => {
            // TODO: these errors are file system errors; make sure an update() will
            // retry this and not cache the file system error, which may be transient.
            return sema.fail(block, operand_src, "unable to open '{s}': {s}", .{ operand, @errorName(err) });
        },
    };
    try mod.semaFile(result.file);
    const file_root_decl = result.file.root_decl.?;
    try mod.declareDeclDependency(sema.owner_decl, file_root_decl);
    return sema.addConstant(file_root_decl.ty, file_root_decl.val);
}

fn zirEmbedFile(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const mod = sema.mod;
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const name = try sema.resolveConstString(block, operand_src, inst_data.operand);

    const embed_file = mod.embedFile(block.getFileScope(), name) catch |err| switch (err) {
        error.ImportOutsidePkgPath => {
            return sema.fail(block, operand_src, "embed of file outside package path: '{s}'", .{name});
        },
        else => {
            // TODO: these errors are file system errors; make sure an update() will
            // retry this and not cache the file system error, which may be transient.
            return sema.fail(block, operand_src, "unable to open '{s}': {s}", .{ name, @errorName(err) });
        },
    };

    var anon_decl = try block.startAnonDecl(LazySrcLoc.unneeded);
    defer anon_decl.deinit();

    const bytes_including_null = embed_file.bytes[0 .. embed_file.bytes.len + 1];

    // TODO instead of using `Value.Tag.bytes`, create a new value tag for pointing at
    // a `*Module.EmbedFile`. The purpose of this would be:
    // - If only the length is read and the bytes are not inspected by comptime code,
    //   there can be an optimization where the codegen backend does a copy_file_range
    //   into the final binary, and never loads the data into memory.
    // - When a Decl is destroyed, it can free the `*Module.EmbedFile`.
    embed_file.owner_decl = try anon_decl.finish(
        try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), embed_file.bytes.len),
        try Value.Tag.bytes.create(anon_decl.arena(), bytes_including_null),
        0, // default alignment
    );

    return sema.analyzeDeclRef(embed_file.owner_decl);
}

fn zirRetErrValueCode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    _ = inst;
    return sema.fail(block, sema.src, "TODO implement zirRetErrValueCode", .{});
}

fn zirShl(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    const target = sema.mod.getTarget();
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);

    const scalar_ty = lhs_ty.scalarType();
    const scalar_rhs_ty = rhs_ty.scalarType();

    // TODO coerce rhs if air_tag is not shl_sat
    const rhs_is_comptime_int = try sema.checkIntType(block, rhs_src, scalar_rhs_ty);

    const maybe_lhs_val = try sema.resolveMaybeUndefVal(block, lhs_src, lhs);
    const maybe_rhs_val = try sema.resolveMaybeUndefVal(block, rhs_src, rhs);

    if (maybe_rhs_val) |rhs_val| {
        if (rhs_val.isUndef()) {
            return sema.addConstUndef(sema.typeOf(lhs));
        }
        if (rhs_val.compareWithZero(.eq)) {
            return lhs;
        }
    }

    const runtime_src = if (maybe_lhs_val) |lhs_val| rs: {
        if (lhs_val.isUndef()) return sema.addConstUndef(lhs_ty);
        const rhs_val = maybe_rhs_val orelse break :rs rhs_src;

        const val = switch (air_tag) {
            .shl_exact => val: {
                const shifted = try lhs_val.shl(rhs_val, lhs_ty, sema.arena, target);
                if (scalar_ty.zigTypeTag() == .ComptimeInt) {
                    break :val shifted;
                }
                const int_info = scalar_ty.intInfo(target);
                const truncated = try shifted.intTrunc(lhs_ty, sema.arena, int_info.signedness, int_info.bits, target);
                if (truncated.compare(.eq, shifted, lhs_ty, target)) {
                    break :val shifted;
                }
                return sema.addConstUndef(lhs_ty);
            },

            .shl_sat => if (scalar_ty.zigTypeTag() == .ComptimeInt)
                try lhs_val.shl(rhs_val, lhs_ty, sema.arena, target)
            else
                try lhs_val.shlSat(rhs_val, lhs_ty, sema.arena, target),

            .shl => if (scalar_ty.zigTypeTag() == .ComptimeInt)
                try lhs_val.shl(rhs_val, lhs_ty, sema.arena, target)
            else
                try lhs_val.shlTrunc(rhs_val, lhs_ty, sema.arena, target),

            else => unreachable,
        };

        return sema.addConstant(lhs_ty, val);
    } else lhs_src;

    // TODO: insert runtime safety check for shl_exact

    const new_rhs = if (air_tag == .shl_sat) rhs: {
        // Limit the RHS type for saturating shl to be an integer as small as the LHS.
        if (rhs_is_comptime_int or
            scalar_rhs_ty.intInfo(target).bits > scalar_ty.intInfo(target).bits)
        {
            const max_int = try sema.addConstant(
                lhs_ty,
                try lhs_ty.maxInt(sema.arena, target),
            );
            const rhs_limited = try sema.analyzeMinMax(block, rhs_src, rhs, max_int, .min, rhs_src, rhs_src);
            break :rhs try sema.intCast(block, lhs_ty, rhs_src, rhs_limited, rhs_src, false);
        } else {
            break :rhs rhs;
        }
    } else rhs;

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addBinOp(air_tag, lhs, new_rhs);
}

fn zirShr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);
    const target = sema.mod.getTarget();

    const runtime_src = if (try sema.resolveMaybeUndefVal(block, rhs_src, rhs)) |rhs_val| rs: {
        if (try sema.resolveMaybeUndefVal(block, lhs_src, lhs)) |lhs_val| {
            if (lhs_val.isUndef() or rhs_val.isUndef()) {
                return sema.addConstUndef(lhs_ty);
            }
            // If rhs is 0, return lhs without doing any calculations.
            if (rhs_val.compareWithZero(.eq)) {
                return sema.addConstant(lhs_ty, lhs_val);
            }
            if (air_tag == .shr_exact) {
                // Detect if any ones would be shifted out.
                const truncated = try lhs_val.intTruncBitsAsValue(lhs_ty, sema.arena, .unsigned, rhs_val, target);
                if (!truncated.compareWithZero(.eq)) {
                    return sema.addConstUndef(lhs_ty);
                }
            }
            const val = try lhs_val.shr(rhs_val, lhs_ty, sema.arena, target);
            return sema.addConstant(lhs_ty, val);
        } else {
            // Even if lhs is not comptime known, we can still deduce certain things based
            // on rhs.
            // If rhs is 0, return lhs without doing any calculations.
            if (rhs_val.compareWithZero(.eq)) {
                return lhs;
            }
            break :rs lhs_src;
        }
    } else rhs_src;

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addBinOp(air_tag, lhs, rhs);
}

fn zirBitwise(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src: LazySrcLoc = .{ .node_offset_bin_op = inst_data.src_node };
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);

    const instructions = &[_]Air.Inst.Ref{ lhs, rhs };
    const resolved_type = try sema.resolvePeerTypes(block, src, instructions, .{ .override = &[_]LazySrcLoc{ lhs_src, rhs_src } });
    const scalar_type = resolved_type.scalarType();
    const scalar_tag = scalar_type.zigTypeTag();

    const casted_lhs = try sema.coerce(block, resolved_type, lhs, lhs_src);
    const casted_rhs = try sema.coerce(block, resolved_type, rhs, rhs_src);

    const is_int = scalar_tag == .Int or scalar_tag == .ComptimeInt;
    const target = sema.mod.getTarget();

    if (!is_int) {
        return sema.fail(block, src, "invalid operands to binary bitwise expression: '{s}' and '{s}'", .{ @tagName(lhs_ty.zigTypeTag()), @tagName(rhs_ty.zigTypeTag()) });
    }

    if (try sema.resolveMaybeUndefVal(block, lhs_src, casted_lhs)) |lhs_val| {
        if (try sema.resolveMaybeUndefVal(block, rhs_src, casted_rhs)) |rhs_val| {
            const result_val = switch (air_tag) {
                .bit_and => try lhs_val.bitwiseAnd(rhs_val, resolved_type, sema.arena, target),
                .bit_or => try lhs_val.bitwiseOr(rhs_val, resolved_type, sema.arena, target),
                .xor => try lhs_val.bitwiseXor(rhs_val, resolved_type, sema.arena, target),
                else => unreachable,
            };
            return sema.addConstant(resolved_type, result_val);
        }
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addBinOp(air_tag, casted_lhs, casted_rhs);
}

fn zirBitNot(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src = src; // TODO put this on the operand, not the '~'

    const operand = sema.resolveInst(inst_data.operand);
    const operand_type = sema.typeOf(operand);
    const scalar_type = operand_type.scalarType();
    const target = sema.mod.getTarget();

    if (scalar_type.zigTypeTag() != .Int) {
        return sema.fail(block, src, "unable to perform binary not operation on type '{}'", .{
            operand_type.fmt(target),
        });
    }

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        if (val.isUndef()) {
            return sema.addConstUndef(operand_type);
        } else if (operand_type.zigTypeTag() == .Vector) {
            const vec_len = try sema.usizeCast(block, operand_src, operand_type.vectorLen());
            var elem_val_buf: Value.ElemValueBuffer = undefined;
            const elems = try sema.arena.alloc(Value, vec_len);
            for (elems) |*elem, i| {
                const elem_val = val.elemValueBuffer(i, &elem_val_buf);
                elem.* = try elem_val.bitwiseNot(scalar_type, sema.arena, target);
            }
            return sema.addConstant(
                operand_type,
                try Value.Tag.aggregate.create(sema.arena, elems),
            );
        } else {
            const result_val = try val.bitwiseNot(operand_type, sema.arena, target);
            return sema.addConstant(operand_type, result_val);
        }
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.not, operand_type, operand);
}

fn analyzeTupleCat(
    sema: *Sema,
    block: *Block,
    src_node: i32,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = src_node };

    const lhs_tuple = lhs_ty.tupleFields();
    const rhs_tuple = rhs_ty.tupleFields();
    const dest_fields = lhs_tuple.types.len + rhs_tuple.types.len;

    if (dest_fields == 0) {
        return sema.addConstant(Type.initTag(.empty_struct_literal), Value.initTag(.empty_struct_value));
    }
    const final_len = try sema.usizeCast(block, rhs_src, dest_fields);

    const types = try sema.arena.alloc(Type, final_len);
    const values = try sema.arena.alloc(Value, final_len);

    const opt_runtime_src = rs: {
        var runtime_src: ?LazySrcLoc = null;
        for (lhs_tuple.types) |ty, i| {
            types[i] = ty;
            values[i] = lhs_tuple.values[i];
            const operand_src = lhs_src; // TODO better source location
            if (values[i].tag() == .unreachable_value) {
                runtime_src = operand_src;
            }
        }
        const offset = lhs_tuple.types.len;
        for (rhs_tuple.types) |ty, i| {
            types[i + offset] = ty;
            values[i + offset] = rhs_tuple.values[i];
            const operand_src = rhs_src; // TODO better source location
            if (rhs_tuple.values[i].tag() == .unreachable_value) {
                runtime_src = operand_src;
            }
        }
        break :rs runtime_src;
    };

    const tuple_ty = try Type.Tag.tuple.create(sema.arena, .{
        .types = types,
        .values = values,
    });

    const runtime_src = opt_runtime_src orelse {
        const tuple_val = try Value.Tag.aggregate.create(sema.arena, values);
        return sema.addConstant(tuple_ty, tuple_val);
    };

    try sema.requireRuntimeBlock(block, runtime_src);

    const element_refs = try sema.arena.alloc(Air.Inst.Ref, final_len);
    for (lhs_tuple.types) |_, i| {
        const operand_src = lhs_src; // TODO better source location
        element_refs[i] = try sema.tupleFieldValByIndex(block, operand_src, lhs, @intCast(u32, i), lhs_ty);
    }
    const offset = lhs_tuple.types.len;
    for (rhs_tuple.types) |_, i| {
        const operand_src = rhs_src; // TODO better source location
        element_refs[i + offset] =
            try sema.tupleFieldValByIndex(block, operand_src, rhs, @intCast(u32, i), rhs_ty);
    }

    return block.addAggregateInit(tuple_ty, element_refs);
}

fn zirArrayCat(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);

    if (lhs_ty.isTuple() and rhs_ty.isTuple()) {
        return sema.analyzeTupleCat(block, inst_data.src_node, lhs, rhs);
    }

    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };

    const target = sema.mod.getTarget();
    const lhs_info = (try sema.getArrayCatInfo(block, lhs_src, lhs)) orelse
        return sema.fail(block, lhs_src, "expected array, found '{}'", .{lhs_ty.fmt(target)});
    const rhs_info = (try sema.getArrayCatInfo(block, rhs_src, rhs)) orelse
        return sema.fail(block, rhs_src, "expected array, found '{}'", .{rhs_ty.fmt(target)});
    if (!lhs_info.elem_type.eql(rhs_info.elem_type, target)) {
        return sema.fail(block, rhs_src, "expected array of type '{}', found '{}'", .{
            lhs_info.elem_type.fmt(target), rhs_ty.fmt(target),
        });
    }

    // When there is a sentinel mismatch, no sentinel on the result. The type system
    // will catch this if it is a problem.
    var res_sent: ?Value = null;
    if (rhs_info.sentinel != null and lhs_info.sentinel != null) {
        if (rhs_info.sentinel.?.eql(lhs_info.sentinel.?, lhs_info.elem_type, target)) {
            res_sent = lhs_info.sentinel.?;
        }
    }

    if (try sema.resolveDefinedValue(block, lhs_src, lhs)) |lhs_val| {
        if (try sema.resolveDefinedValue(block, rhs_src, rhs)) |rhs_val| {
            const lhs_len = try sema.usizeCast(block, lhs_src, lhs_info.len);
            const rhs_len = try sema.usizeCast(block, lhs_src, rhs_info.len);
            const final_len = lhs_len + rhs_len;
            const final_len_including_sent = final_len + @boolToInt(res_sent != null);
            const lhs_single_ptr = lhs_ty.zigTypeTag() == .Pointer and !lhs_ty.isSlice();
            const rhs_single_ptr = rhs_ty.zigTypeTag() == .Pointer and !rhs_ty.isSlice();
            const lhs_sub_val = if (lhs_single_ptr) (try sema.pointerDeref(block, lhs_src, lhs_val, lhs_ty)).? else lhs_val;
            const rhs_sub_val = if (rhs_single_ptr) (try sema.pointerDeref(block, rhs_src, rhs_val, rhs_ty)).? else rhs_val;
            var anon_decl = try block.startAnonDecl(LazySrcLoc.unneeded);
            defer anon_decl.deinit();

            const buf = try anon_decl.arena().alloc(Value, final_len_including_sent);
            {
                var i: usize = 0;
                while (i < lhs_len) : (i += 1) {
                    const val = try lhs_sub_val.elemValue(sema.arena, i);
                    buf[i] = try val.copy(anon_decl.arena());
                }
            }
            {
                var i: usize = 0;
                while (i < rhs_len) : (i += 1) {
                    const val = try rhs_sub_val.elemValue(sema.arena, i);
                    buf[lhs_len + i] = try val.copy(anon_decl.arena());
                }
            }
            const ty = if (res_sent) |rs| ty: {
                buf[final_len] = try rs.copy(anon_decl.arena());
                break :ty try Type.Tag.array_sentinel.create(anon_decl.arena(), .{
                    .len = final_len,
                    .elem_type = try lhs_info.elem_type.copy(anon_decl.arena()),
                    .sentinel = try rs.copy(anon_decl.arena()),
                });
            } else try Type.Tag.array.create(anon_decl.arena(), .{
                .len = final_len,
                .elem_type = try lhs_info.elem_type.copy(anon_decl.arena()),
            });
            const val = try Value.Tag.aggregate.create(anon_decl.arena(), buf);
            const decl = try anon_decl.finish(ty, val, 0);
            if (lhs_ty.zigTypeTag() == .Pointer or rhs_ty.zigTypeTag() == .Pointer) {
                return sema.analyzeDeclRef(decl);
            } else {
                return sema.analyzeDeclVal(block, .unneeded, decl);
            }
        } else {
            return sema.fail(block, lhs_src, "TODO runtime array_cat", .{});
        }
    } else {
        return sema.fail(block, lhs_src, "TODO runtime array_cat", .{});
    }
}

fn getArrayCatInfo(sema: *Sema, block: *Block, src: LazySrcLoc, inst: Air.Inst.Ref) !?Type.ArrayInfo {
    const t = sema.typeOf(inst);
    const target = sema.mod.getTarget();
    return switch (t.zigTypeTag()) {
        .Array => t.arrayInfo(),
        .Pointer => blk: {
            const ptrinfo = t.ptrInfo().data;
            if (ptrinfo.size == .Slice) {
                const val = try sema.resolveConstValue(block, src, inst);
                return Type.ArrayInfo{
                    .elem_type = t.childType(),
                    .sentinel = t.sentinel(),
                    .len = val.sliceLen(target),
                };
            }
            if (ptrinfo.pointee_type.zigTypeTag() != .Array) return null;
            if (ptrinfo.size != .One) return null;
            break :blk ptrinfo.pointee_type.arrayInfo();
        },
        else => null,
    };
}

fn analyzeTupleMul(
    sema: *Sema,
    block: *Block,
    src_node: i32,
    operand: Air.Inst.Ref,
    factor: u64,
) CompileError!Air.Inst.Ref {
    const operand_ty = sema.typeOf(operand);
    const operand_tuple = operand_ty.tupleFields();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = src_node };

    const tuple_len = operand_tuple.types.len;
    const final_len_u64 = std.math.mul(u64, tuple_len, factor) catch
        return sema.fail(block, rhs_src, "operation results in overflow", .{});

    if (final_len_u64 == 0) {
        return sema.addConstant(Type.initTag(.empty_struct_literal), Value.initTag(.empty_struct_value));
    }
    const final_len = try sema.usizeCast(block, rhs_src, final_len_u64);

    const types = try sema.arena.alloc(Type, final_len);
    const values = try sema.arena.alloc(Value, final_len);

    const opt_runtime_src = rs: {
        var runtime_src: ?LazySrcLoc = null;
        for (operand_tuple.types) |ty, i| {
            types[i] = ty;
            values[i] = operand_tuple.values[i];
            const operand_src = lhs_src; // TODO better source location
            if (values[i].tag() == .unreachable_value) {
                runtime_src = operand_src;
            }
        }
        var i: usize = 1;
        while (i < factor) : (i += 1) {
            mem.copy(Type, types[tuple_len * i ..], operand_tuple.types);
            mem.copy(Value, values[tuple_len * i ..], operand_tuple.values);
        }
        break :rs runtime_src;
    };

    const tuple_ty = try Type.Tag.tuple.create(sema.arena, .{
        .types = types,
        .values = values,
    });

    const runtime_src = opt_runtime_src orelse {
        const tuple_val = try Value.Tag.aggregate.create(sema.arena, values);
        return sema.addConstant(tuple_ty, tuple_val);
    };

    try sema.requireRuntimeBlock(block, runtime_src);

    const element_refs = try sema.arena.alloc(Air.Inst.Ref, final_len);
    for (operand_tuple.types) |_, i| {
        const operand_src = lhs_src; // TODO better source location
        element_refs[i] = try sema.tupleFieldValByIndex(block, operand_src, operand, @intCast(u32, i), operand_ty);
    }
    var i: usize = 1;
    while (i < factor) : (i += 1) {
        mem.copy(Air.Inst.Ref, element_refs[tuple_len * i ..], element_refs[0..tuple_len]);
    }

    return block.addAggregateInit(tuple_ty, element_refs);
}

fn zirArrayMul(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const lhs_ty = sema.typeOf(lhs);
    const src: LazySrcLoc = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };

    // In `**` rhs has to be comptime-known, but lhs can be runtime-known
    const factor = try sema.resolveInt(block, rhs_src, extra.rhs, Type.usize);

    if (lhs_ty.isTuple()) {
        return sema.analyzeTupleMul(block, inst_data.src_node, lhs, factor);
    }
    const target = sema.mod.getTarget();

    const mulinfo = (try sema.getArrayCatInfo(block, lhs_src, lhs)) orelse
        return sema.fail(block, lhs_src, "expected array, found '{}'", .{lhs_ty.fmt(target)});

    const final_len_u64 = std.math.mul(u64, mulinfo.len, factor) catch
        return sema.fail(block, rhs_src, "operation results in overflow", .{});

    if (try sema.resolveDefinedValue(block, lhs_src, lhs)) |lhs_val| {
        const final_len = try sema.usizeCast(block, src, final_len_u64);
        const final_len_including_sent = final_len + @boolToInt(mulinfo.sentinel != null);
        const lhs_len = try sema.usizeCast(block, lhs_src, mulinfo.len);

        const is_single_ptr = lhs_ty.zigTypeTag() == .Pointer and !lhs_ty.isSlice();
        const lhs_sub_val = if (is_single_ptr) (try sema.pointerDeref(block, lhs_src, lhs_val, lhs_ty)).? else lhs_val;

        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();

        const final_ty = if (mulinfo.sentinel) |sent|
            try Type.Tag.array_sentinel.create(anon_decl.arena(), .{
                .len = final_len,
                .elem_type = try mulinfo.elem_type.copy(anon_decl.arena()),
                .sentinel = try sent.copy(anon_decl.arena()),
            })
        else
            try Type.Tag.array.create(anon_decl.arena(), .{
                .len = final_len,
                .elem_type = try mulinfo.elem_type.copy(anon_decl.arena()),
            });
        const buf = try anon_decl.arena().alloc(Value, final_len_including_sent);

        // Optimization for the common pattern of a single element repeated N times, such
        // as zero-filling a byte array.
        const val = if (lhs_len == 1) blk: {
            const elem_val = try lhs_sub_val.elemValue(sema.arena, 0);
            const copied_val = try elem_val.copy(anon_decl.arena());
            break :blk try Value.Tag.repeated.create(anon_decl.arena(), copied_val);
        } else blk: {
            // the actual loop
            var i: usize = 0;
            while (i < factor) : (i += 1) {
                var j: usize = 0;
                while (j < lhs_len) : (j += 1) {
                    const val = try lhs_sub_val.elemValue(sema.arena, j);
                    buf[lhs_len * i + j] = try val.copy(anon_decl.arena());
                }
            }
            if (mulinfo.sentinel) |sent| {
                buf[final_len] = try sent.copy(anon_decl.arena());
            }
            break :blk try Value.Tag.aggregate.create(anon_decl.arena(), buf);
        };
        const decl = try anon_decl.finish(final_ty, val, 0);
        if (lhs_ty.zigTypeTag() == .Pointer) {
            return sema.analyzeDeclRef(decl);
        } else {
            return sema.analyzeDeclVal(block, .unneeded, decl);
        }
    }
    return sema.fail(block, lhs_src, "TODO runtime array_mul", .{});
}

fn zirNegate(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    tag_override: Zir.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const lhs_src = src;
    const rhs_src = src; // TODO better source location

    const rhs = sema.resolveInst(inst_data.operand);
    const rhs_ty = sema.typeOf(rhs);
    const rhs_scalar_ty = rhs_ty.scalarType();

    const target = sema.mod.getTarget();
    if (tag_override == .sub and rhs_scalar_ty.isUnsignedInt()) {
        return sema.fail(block, src, "negation of type '{}'", .{rhs_ty.fmt(target)});
    }

    const lhs = if (rhs_ty.zigTypeTag() == .Vector)
        try sema.addConstant(rhs_ty, try Value.Tag.repeated.create(sema.arena, Value.zero))
    else
        sema.resolveInst(.zero);

    return sema.analyzeArithmetic(block, tag_override, lhs, rhs, src, lhs_src, rhs_src);
}

fn zirArithmetic(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    zir_tag: Zir.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    sema.src = .{ .node_offset_bin_op = inst_data.src_node };
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);

    return sema.analyzeArithmetic(block, zir_tag, lhs, rhs, sema.src, lhs_src, rhs_src);
}

fn zirOverflowArithmetic(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
    zir_tag: Zir.Inst.Extended,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const extra = sema.code.extraData(Zir.Inst.OverflowArithmetic, extended.operand).data;
    const src: LazySrcLoc = .{ .node_offset = extra.node };

    const lhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const rhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = extra.node };
    const ptr_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = extra.node };

    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const ptr = sema.resolveInst(extra.ptr);

    const lhs_ty = sema.typeOf(lhs);
    const target = sema.mod.getTarget();

    // Note, the types of lhs/rhs (also for shifting)/ptr are already correct as ensured by astgen.
    const dest_ty = lhs_ty;
    if (dest_ty.zigTypeTag() != .Int) {
        return sema.fail(block, src, "expected integer type, found '{}'", .{dest_ty.fmt(target)});
    }

    const maybe_lhs_val = try sema.resolveMaybeUndefVal(block, lhs_src, lhs);
    const maybe_rhs_val = try sema.resolveMaybeUndefVal(block, rhs_src, rhs);

    const types = try sema.arena.alloc(Type, 2);
    const values = try sema.arena.alloc(Value, 2);
    const tuple_ty = try Type.Tag.tuple.create(sema.arena, .{
        .types = types,
        .values = values,
    });

    types[0] = dest_ty;
    types[1] = Type.initTag(.u1);
    values[0] = Value.initTag(.unreachable_value);
    values[1] = Value.initTag(.unreachable_value);

    const result: struct {
        overflowed: enum { yes, no, undef },
        wrapped: Air.Inst.Ref,
    } = result: {
        switch (zir_tag) {
            .add_with_overflow => {
                // If either of the arguments is zero, `false` is returned and the other is stored
                // to the result, even if it is undefined..
                // Otherwise, if either of the argument is undefined, undefined is returned.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef() and lhs_val.compareWithZero(.eq)) {
                        break :result .{ .overflowed = .no, .wrapped = rhs };
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (!rhs_val.isUndef() and rhs_val.compareWithZero(.eq)) {
                        break :result .{ .overflowed = .no, .wrapped = lhs };
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (maybe_rhs_val) |rhs_val| {
                        if (lhs_val.isUndef() or rhs_val.isUndef()) {
                            break :result .{ .overflowed = .undef, .wrapped = try sema.addConstUndef(dest_ty) };
                        }

                        const result = try lhs_val.intAddWithOverflow(rhs_val, dest_ty, sema.arena, target);
                        const inst = try sema.addConstant(dest_ty, result.wrapped_result);
                        break :result .{ .overflowed = if (result.overflowed) .yes else .no, .wrapped = inst };
                    }
                }
            },
            .sub_with_overflow => {
                // If the rhs is zero, then the result is lhs and no overflow occured.
                // Otherwise, if either result is undefined, both results are undefined.
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        break :result .{ .overflowed = .undef, .wrapped = try sema.addConstUndef(dest_ty) };
                    } else if (rhs_val.compareWithZero(.eq)) {
                        break :result .{ .overflowed = .no, .wrapped = lhs };
                    } else if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            break :result .{ .overflowed = .undef, .wrapped = try sema.addConstUndef(dest_ty) };
                        }

                        const result = try lhs_val.intSubWithOverflow(rhs_val, dest_ty, sema.arena, target);
                        const inst = try sema.addConstant(dest_ty, result.wrapped_result);
                        break :result .{ .overflowed = if (result.overflowed) .yes else .no, .wrapped = inst };
                    }
                }
            },
            .mul_with_overflow => {
                // If either of the arguments is zero, the result is zero and no overflow occured.
                // If either of the arguments is one, the result is the other and no overflow occured.
                // Otherwise, if either of the arguments is undefined, both results are undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            break :result .{ .overflowed = .no, .wrapped = lhs };
                        } else if (lhs_val.compare(.eq, Value.one, dest_ty, target)) {
                            break :result .{ .overflowed = .no, .wrapped = rhs };
                        }
                    }
                }

                if (maybe_rhs_val) |rhs_val| {
                    if (!rhs_val.isUndef()) {
                        if (rhs_val.compareWithZero(.eq)) {
                            break :result .{ .overflowed = .no, .wrapped = rhs };
                        } else if (rhs_val.compare(.eq, Value.one, dest_ty, target)) {
                            break :result .{ .overflowed = .no, .wrapped = lhs };
                        }
                    }
                }

                if (maybe_lhs_val) |lhs_val| {
                    if (maybe_rhs_val) |rhs_val| {
                        if (lhs_val.isUndef() or rhs_val.isUndef()) {
                            break :result .{ .overflowed = .undef, .wrapped = try sema.addConstUndef(dest_ty) };
                        }

                        const result = try lhs_val.intMulWithOverflow(rhs_val, dest_ty, sema.arena, target);
                        const inst = try sema.addConstant(dest_ty, result.wrapped_result);
                        break :result .{ .overflowed = if (result.overflowed) .yes else .no, .wrapped = inst };
                    }
                }
            },
            .shl_with_overflow => {
                // If lhs is zero, the result is zero and no overflow occurred.
                // If rhs is zero, the result is lhs (even if undefined) and no overflow occurred.
                // Oterhwise if either of the arguments is undefined, both results are undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef() and lhs_val.compareWithZero(.eq)) {
                        break :result .{ .overflowed = .no, .wrapped = lhs };
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (!rhs_val.isUndef() and rhs_val.compareWithZero(.eq)) {
                        break :result .{ .overflowed = .no, .wrapped = lhs };
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (maybe_rhs_val) |rhs_val| {
                        if (lhs_val.isUndef() or rhs_val.isUndef()) {
                            break :result .{ .overflowed = .undef, .wrapped = try sema.addConstUndef(dest_ty) };
                        }

                        const result = try lhs_val.shlWithOverflow(rhs_val, dest_ty, sema.arena, target);
                        const inst = try sema.addConstant(dest_ty, result.wrapped_result);
                        break :result .{ .overflowed = if (result.overflowed) .yes else .no, .wrapped = inst };
                    }
                }
            },
            else => unreachable,
        }

        const air_tag: Air.Inst.Tag = switch (zir_tag) {
            .add_with_overflow => .add_with_overflow,
            .mul_with_overflow => .mul_with_overflow,
            .sub_with_overflow => .sub_with_overflow,
            .shl_with_overflow => .shl_with_overflow,
            else => unreachable,
        };

        try sema.requireRuntimeBlock(block, src);

        const tuple = try block.addInst(.{
            .tag = air_tag,
            .data = .{ .ty_pl = .{
                .ty = try block.sema.addType(tuple_ty),
                .payload = try block.sema.addExtra(Air.Bin{
                    .lhs = lhs,
                    .rhs = rhs,
                }),
            } },
        });

        const wrapped = try block.addStructFieldVal(tuple, 0, dest_ty);
        try sema.storePtr2(block, src, ptr, ptr_src, wrapped, src, .store);

        const overflow_bit = try block.addStructFieldVal(tuple, 1, Type.initTag(.u1));
        const zero_u1 = try sema.addConstant(Type.initTag(.u1), Value.zero);
        return try block.addBinOp(.cmp_neq, overflow_bit, zero_u1);
    };

    try sema.storePtr2(block, src, ptr, ptr_src, result.wrapped, src, .store);

    return switch (result.overflowed) {
        .yes => Air.Inst.Ref.bool_true,
        .no => Air.Inst.Ref.bool_false,
        .undef => try sema.addConstUndef(Type.bool),
    };
}

fn analyzeArithmetic(
    sema: *Sema,
    block: *Block,
    /// TODO performance investigation: make this comptime?
    zir_tag: Zir.Inst.Tag,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
    src: LazySrcLoc,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    const lhs_zig_ty_tag = try lhs_ty.zigTypeTagOrPoison();
    const rhs_zig_ty_tag = try rhs_ty.zigTypeTagOrPoison();
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);

    if (lhs_zig_ty_tag == .Pointer) switch (lhs_ty.ptrSize()) {
        .One, .Slice => {},
        .Many, .C => {
            const op_src = src; // TODO better source location
            const air_tag: Air.Inst.Tag = switch (zir_tag) {
                .add => .ptr_add,
                .sub => .ptr_sub,
                else => return sema.fail(
                    block,
                    op_src,
                    "invalid pointer arithmetic operand: '{s}''",
                    .{@tagName(zir_tag)},
                ),
            };
            return analyzePtrArithmetic(sema, block, op_src, lhs, rhs, air_tag, lhs_src, rhs_src);
        },
    };

    const instructions = &[_]Air.Inst.Ref{ lhs, rhs };
    const resolved_type = try sema.resolvePeerTypes(block, src, instructions, .{
        .override = &[_]LazySrcLoc{ lhs_src, rhs_src },
    });

    const casted_lhs = try sema.coerce(block, resolved_type, lhs, lhs_src);
    const casted_rhs = try sema.coerce(block, resolved_type, rhs, rhs_src);

    const lhs_scalar_ty = lhs_ty.scalarType();
    const rhs_scalar_ty = rhs_ty.scalarType();
    const scalar_tag = resolved_type.scalarType().zigTypeTag();

    const is_int = scalar_tag == .Int or scalar_tag == .ComptimeInt;
    const is_float = scalar_tag == .Float or scalar_tag == .ComptimeFloat;

    if (!is_int and !(is_float and floatOpAllowed(zir_tag))) {
        return sema.fail(block, src, "invalid operands to binary expression: '{s}' and '{s}'", .{
            @tagName(lhs_zig_ty_tag), @tagName(rhs_zig_ty_tag),
        });
    }

    const target = sema.mod.getTarget();
    const maybe_lhs_val = try sema.resolveMaybeUndefVal(block, lhs_src, casted_lhs);
    const maybe_rhs_val = try sema.resolveMaybeUndefVal(block, rhs_src, casted_rhs);
    const rs: struct { src: LazySrcLoc, air_tag: Air.Inst.Tag } = rs: {
        switch (zir_tag) {
            .add => {
                // For integers:
                // If either of the operands are zero, then the other operand is
                // returned, even if it is undefined.
                // If either of the operands are undefined, it's a compile error
                // because there is a possible value for which the addition would
                // overflow (max_int), causing illegal behavior.
                // For floats: either operand being undef makes the result undef.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef() and lhs_val.compareWithZero(.eq)) {
                        return casted_rhs;
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        if (is_int) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        } else {
                            return sema.addConstUndef(resolved_type);
                        }
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        if (is_int) {
                            return sema.failWithUseOfUndef(block, lhs_src);
                        } else {
                            return sema.addConstUndef(resolved_type);
                        }
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intAdd(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatAdd(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = rhs_src, .air_tag = .add };
                } else break :rs .{ .src = lhs_src, .air_tag = .add };
            },
            .addwrap => {
                // Integers only; floats are checked above.
                // If either of the operands are zero, the other operand is returned.
                // If either of the operands are undefined, the result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef() and lhs_val.compareWithZero(.eq)) {
                        return casted_rhs;
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.numberAddWrap(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else break :rs .{ .src = lhs_src, .air_tag = .addwrap };
                } else break :rs .{ .src = rhs_src, .air_tag = .addwrap };
            },
            .add_sat => {
                // Integers only; floats are checked above.
                // If either of the operands are zero, then the other operand is returned.
                // If either of the operands are undefined, the result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef() and lhs_val.compareWithZero(.eq)) {
                        return casted_rhs;
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        const val = if (scalar_tag == .ComptimeInt)
                            try lhs_val.intAdd(rhs_val, resolved_type, sema.arena, target)
                        else
                            try lhs_val.intAddSat(rhs_val, resolved_type, sema.arena, target);

                        return sema.addConstant(resolved_type, val);
                    } else break :rs .{ .src = lhs_src, .air_tag = .add_sat };
                } else break :rs .{ .src = rhs_src, .air_tag = .add_sat };
            },
            .sub => {
                // For integers:
                // If the rhs is zero, then the other operand is
                // returned, even if it is undefined.
                // If either of the operands are undefined, it's a compile error
                // because there is a possible value for which the subtraction would
                // overflow, causing illegal behavior.
                // For floats: either operand being undef makes the result undef.
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        if (is_int) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        } else {
                            return sema.addConstUndef(resolved_type);
                        }
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        if (is_int) {
                            return sema.failWithUseOfUndef(block, lhs_src);
                        } else {
                            return sema.addConstUndef(resolved_type);
                        }
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intSub(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatSub(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = rhs_src, .air_tag = .sub };
                } else break :rs .{ .src = lhs_src, .air_tag = .sub };
            },
            .subwrap => {
                // Integers only; floats are checked above.
                // If the RHS is zero, then the other operand is returned, even if it is undefined.
                // If either of the operands are undefined, the result is undefined.
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.numberSubWrap(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else break :rs .{ .src = rhs_src, .air_tag = .subwrap };
                } else break :rs .{ .src = lhs_src, .air_tag = .subwrap };
            },
            .sub_sat => {
                // Integers only; floats are checked above.
                // If the RHS is zero, result is LHS.
                // If either of the operands are undefined, result is undefined.
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return casted_lhs;
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        const val = if (scalar_tag == .ComptimeInt)
                            try lhs_val.intSub(rhs_val, resolved_type, sema.arena, target)
                        else
                            try lhs_val.intSubSat(rhs_val, resolved_type, sema.arena, target);

                        return sema.addConstant(resolved_type, val);
                    } else break :rs .{ .src = rhs_src, .air_tag = .sub_sat };
                } else break :rs .{ .src = lhs_src, .air_tag = .sub_sat };
            },
            .div => {
                // TODO: emit compile error when .div is used on integers and there would be an
                // ambiguous result between div_floor and div_trunc.

                // For integers:
                // If the lhs is zero, then zero is returned regardless of rhs.
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined:
                //   * if lhs type is signed:
                //     * if rhs is comptime-known and not -1, result is undefined
                //     * if rhs is -1 or runtime-known, compile error because there is a
                //        possible value (-min_int / -1)  for which division would be
                //        illegal behavior.
                //   * if lhs type is unsigned, undef is returned regardless of rhs.
                // TODO: emit runtime safety for division by zero
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        if (lhs_scalar_ty.isSignedInt() and rhs_scalar_ty.isSignedInt()) {
                            if (maybe_rhs_val) |rhs_val| {
                                if (rhs_val.compare(.neq, Value.negative_one, resolved_type, target)) {
                                    return sema.addConstUndef(resolved_type);
                                }
                            }
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        return sema.addConstUndef(resolved_type);
                    }

                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intDiv(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatDiv(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else {
                        if (is_int) {
                            break :rs .{ .src = rhs_src, .air_tag = .div_trunc };
                        } else {
                            break :rs .{ .src = rhs_src, .air_tag = .div_float };
                        }
                    }
                } else {
                    if (is_int) {
                        break :rs .{ .src = lhs_src, .air_tag = .div_trunc };
                    } else {
                        break :rs .{ .src = lhs_src, .air_tag = .div_float };
                    }
                }
            },
            .div_trunc => {
                // For integers:
                // If the lhs is zero, then zero is returned regardless of rhs.
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined:
                //   * if lhs type is signed:
                //     * if rhs is comptime-known and not -1, result is undefined
                //     * if rhs is -1 or runtime-known, compile error because there is a
                //        possible value (-min_int / -1)  for which division would be
                //        illegal behavior.
                //   * if lhs type is unsigned, undef is returned regardless of rhs.
                // TODO: emit runtime safety for division by zero
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        if (lhs_scalar_ty.isSignedInt() and rhs_scalar_ty.isSignedInt()) {
                            if (maybe_rhs_val) |rhs_val| {
                                if (rhs_val.compare(.neq, Value.negative_one, resolved_type, target)) {
                                    return sema.addConstUndef(resolved_type);
                                }
                            }
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        return sema.addConstUndef(resolved_type);
                    }

                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intDiv(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatDivTrunc(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = rhs_src, .air_tag = .div_trunc };
                } else break :rs .{ .src = lhs_src, .air_tag = .div_trunc };
            },
            .div_floor => {
                // For integers:
                // If the lhs is zero, then zero is returned regardless of rhs.
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined:
                //   * if lhs type is signed:
                //     * if rhs is comptime-known and not -1, result is undefined
                //     * if rhs is -1 or runtime-known, compile error because there is a
                //        possible value (-min_int / -1)  for which division would be
                //        illegal behavior.
                //   * if lhs type is unsigned, undef is returned regardless of rhs.
                // TODO: emit runtime safety for division by zero
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        if (lhs_scalar_ty.isSignedInt() and rhs_scalar_ty.isSignedInt()) {
                            if (maybe_rhs_val) |rhs_val| {
                                if (rhs_val.compare(.neq, Value.negative_one, resolved_type, target)) {
                                    return sema.addConstUndef(resolved_type);
                                }
                            }
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        return sema.addConstUndef(resolved_type);
                    }

                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intDivFloor(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatDivFloor(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = rhs_src, .air_tag = .div_floor };
                } else break :rs .{ .src = lhs_src, .air_tag = .div_floor };
            },
            .div_exact => {
                // For integers:
                // If the lhs is zero, then zero is returned regardless of rhs.
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, compile error because there is a possible
                // value for which the division would result in a remainder.
                // TODO: emit runtime safety for if there is a remainder
                // TODO: emit runtime safety for division by zero
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, compile error because there is a possible
                // value for which the division would result in a remainder.
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    } else {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (maybe_rhs_val) |rhs_val| {
                        if (is_int) {
                            // TODO: emit compile error if there is a remainder
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intDiv(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            // TODO: emit compile error if there is a remainder
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatDiv(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = rhs_src, .air_tag = .div_exact };
                } else break :rs .{ .src = lhs_src, .air_tag = .div_exact };
            },
            .mul => {
                // For integers:
                // If either of the operands are zero, the result is zero.
                // If either of the operands are one, the result is the other
                // operand, even if it is undefined.
                // If either of the operands are undefined, it's a compile error
                // because there is a possible value for which the addition would
                // overflow (max_int), causing illegal behavior.
                // For floats: either operand being undef makes the result undef.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                        if (lhs_val.compare(.eq, Value.one, resolved_type, target)) {
                            return casted_rhs;
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        if (is_int) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        } else {
                            return sema.addConstUndef(resolved_type);
                        }
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.addConstant(resolved_type, Value.zero);
                    }
                    if (rhs_val.compare(.eq, Value.one, resolved_type, target)) {
                        return casted_lhs;
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            if (is_int) {
                                return sema.failWithUseOfUndef(block, lhs_src);
                            } else {
                                return sema.addConstUndef(resolved_type);
                            }
                        }
                        if (is_int) {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intMul(rhs_val, resolved_type, sema.arena, target),
                            );
                        } else {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.floatMul(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                    } else break :rs .{ .src = lhs_src, .air_tag = .mul };
                } else break :rs .{ .src = rhs_src, .air_tag = .mul };
            },
            .mulwrap => {
                // Integers only; floats are handled above.
                // If either of the operands are zero, result is zero.
                // If either of the operands are one, result is the other operand.
                // If either of the operands are undefined, result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                        if (lhs_val.compare(.eq, Value.one, resolved_type, target)) {
                            return casted_rhs;
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.addConstant(resolved_type, Value.zero);
                    }
                    if (rhs_val.compare(.eq, Value.one, resolved_type, target)) {
                        return casted_lhs;
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            return sema.addConstUndef(resolved_type);
                        }
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.numberMulWrap(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else break :rs .{ .src = lhs_src, .air_tag = .mulwrap };
                } else break :rs .{ .src = rhs_src, .air_tag = .mulwrap };
            },
            .mul_sat => {
                // Integers only; floats are checked above.
                // If either of the operands are zero, result is zero.
                // If either of the operands are one, result is the other operand.
                // If either of the operands are undefined, result is undefined.
                if (maybe_lhs_val) |lhs_val| {
                    if (!lhs_val.isUndef()) {
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                        if (lhs_val.compare(.eq, Value.one, resolved_type, target)) {
                            return casted_rhs;
                        }
                    }
                }
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.addConstant(resolved_type, Value.zero);
                    }
                    if (rhs_val.compare(.eq, Value.one, resolved_type, target)) {
                        return casted_lhs;
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            return sema.addConstUndef(resolved_type);
                        }

                        const val = if (scalar_tag == .ComptimeInt)
                            try lhs_val.intMul(rhs_val, resolved_type, sema.arena, target)
                        else
                            try lhs_val.intMulSat(rhs_val, resolved_type, sema.arena, target);

                        return sema.addConstant(resolved_type, val);
                    } else break :rs .{ .src = lhs_src, .air_tag = .mul_sat };
                } else break :rs .{ .src = rhs_src, .air_tag = .mul_sat };
            },
            .mod_rem => {
                // For integers:
                // Either operand being undef is a compile error because there exists
                // a possible value (TODO what is it?) that would invoke illegal behavior.
                // TODO: can lhs undef be handled better?
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                //
                // For either one: if the result would be different between @mod and @rem,
                // then emit a compile error saying you have to pick one.
                if (is_int) {
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, lhs_src);
                        }
                        if (lhs_val.compareWithZero(.eq)) {
                            return sema.addConstant(resolved_type, Value.zero);
                        }
                    } else if (lhs_scalar_ty.isSignedInt()) {
                        return sema.failWithModRemNegative(block, lhs_src, lhs_ty, rhs_ty);
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        if (rhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        if (rhs_val.compareWithZero(.eq)) {
                            return sema.failWithDivideByZero(block, rhs_src);
                        }
                        if (maybe_lhs_val) |lhs_val| {
                            const rem_result = try lhs_val.intRem(rhs_val, resolved_type, sema.arena, target);
                            // If this answer could possibly be different by doing `intMod`,
                            // we must emit a compile error. Otherwise, it's OK.
                            if (rhs_val.compareWithZero(.lt) != lhs_val.compareWithZero(.lt) and
                                !rem_result.compareWithZero(.eq))
                            {
                                const bad_src = if (lhs_val.compareWithZero(.lt))
                                    lhs_src
                                else
                                    rhs_src;
                                return sema.failWithModRemNegative(block, bad_src, lhs_ty, rhs_ty);
                            }
                            if (lhs_val.compareWithZero(.lt)) {
                                // Negative
                                return sema.addConstant(resolved_type, Value.zero);
                            }
                            return sema.addConstant(resolved_type, rem_result);
                        }
                        break :rs .{ .src = lhs_src, .air_tag = .rem };
                    } else if (rhs_scalar_ty.isSignedInt()) {
                        return sema.failWithModRemNegative(block, rhs_src, lhs_ty, rhs_ty);
                    } else {
                        break :rs .{ .src = rhs_src, .air_tag = .rem };
                    }
                }
                // float operands
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.lt)) {
                        return sema.failWithModRemNegative(block, rhs_src, lhs_ty, rhs_ty);
                    }
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef() or lhs_val.compareWithZero(.lt)) {
                            return sema.failWithModRemNegative(block, lhs_src, lhs_ty, rhs_ty);
                        }
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.floatRem(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else {
                        return sema.failWithModRemNegative(block, lhs_src, lhs_ty, rhs_ty);
                    }
                } else {
                    return sema.failWithModRemNegative(block, rhs_src, lhs_ty, rhs_ty);
                }
            },
            .rem => {
                // For integers:
                // Either operand being undef is a compile error because there exists
                // a possible value (TODO what is it?) that would invoke illegal behavior.
                // TODO: can lhs zero be handled better?
                // TODO: can lhs undef be handled better?
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                if (is_int) {
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, lhs_src);
                        }
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        if (rhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        if (rhs_val.compareWithZero(.eq)) {
                            return sema.failWithDivideByZero(block, rhs_src);
                        }
                        if (maybe_lhs_val) |lhs_val| {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intRem(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                        break :rs .{ .src = lhs_src, .air_tag = .rem };
                    } else {
                        break :rs .{ .src = rhs_src, .air_tag = .rem };
                    }
                }
                // float operands
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.floatRem(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else break :rs .{ .src = rhs_src, .air_tag = .rem };
                } else break :rs .{ .src = lhs_src, .air_tag = .rem };
            },
            .mod => {
                // For integers:
                // Either operand being undef is a compile error because there exists
                // a possible value (TODO what is it?) that would invoke illegal behavior.
                // TODO: can lhs zero be handled better?
                // TODO: can lhs undef be handled better?
                //
                // For floats:
                // If the rhs is zero, compile error for division by zero.
                // If the rhs is undefined, compile error because there is a possible
                // value (zero) for which the division would be illegal behavior.
                // If the lhs is undefined, result is undefined.
                if (is_int) {
                    if (maybe_lhs_val) |lhs_val| {
                        if (lhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, lhs_src);
                        }
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        if (rhs_val.isUndef()) {
                            return sema.failWithUseOfUndef(block, rhs_src);
                        }
                        if (rhs_val.compareWithZero(.eq)) {
                            return sema.failWithDivideByZero(block, rhs_src);
                        }
                        if (maybe_lhs_val) |lhs_val| {
                            return sema.addConstant(
                                resolved_type,
                                try lhs_val.intMod(rhs_val, resolved_type, sema.arena, target),
                            );
                        }
                        break :rs .{ .src = lhs_src, .air_tag = .mod };
                    } else {
                        break :rs .{ .src = rhs_src, .air_tag = .mod };
                    }
                }
                // float operands
                if (maybe_rhs_val) |rhs_val| {
                    if (rhs_val.isUndef()) {
                        return sema.failWithUseOfUndef(block, rhs_src);
                    }
                    if (rhs_val.compareWithZero(.eq)) {
                        return sema.failWithDivideByZero(block, rhs_src);
                    }
                }
                if (maybe_lhs_val) |lhs_val| {
                    if (lhs_val.isUndef()) {
                        return sema.addConstUndef(resolved_type);
                    }
                    if (maybe_rhs_val) |rhs_val| {
                        return sema.addConstant(
                            resolved_type,
                            try lhs_val.floatMod(rhs_val, resolved_type, sema.arena, target),
                        );
                    } else break :rs .{ .src = rhs_src, .air_tag = .mod };
                } else break :rs .{ .src = lhs_src, .air_tag = .mod };
            },
            else => unreachable,
        }
    };

    try sema.requireRuntimeBlock(block, rs.src);
    return block.addBinOp(rs.air_tag, casted_lhs, casted_rhs);
}

fn analyzePtrArithmetic(
    sema: *Sema,
    block: *Block,
    op_src: LazySrcLoc,
    ptr: Air.Inst.Ref,
    uncasted_offset: Air.Inst.Ref,
    air_tag: Air.Inst.Tag,
    ptr_src: LazySrcLoc,
    offset_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    // TODO if the operand is comptime-known to be negative, or is a negative int,
    // coerce to isize instead of usize.
    const offset = try sema.coerce(block, Type.usize, uncasted_offset, offset_src);
    // TODO adjust the return type according to alignment and other factors
    const target = sema.mod.getTarget();
    const runtime_src = rs: {
        if (try sema.resolveMaybeUndefVal(block, ptr_src, ptr)) |ptr_val| {
            if (try sema.resolveMaybeUndefVal(block, offset_src, offset)) |offset_val| {
                const ptr_ty = sema.typeOf(ptr);
                const new_ptr_ty = ptr_ty; // TODO modify alignment

                if (ptr_val.isUndef() or offset_val.isUndef()) {
                    return sema.addConstUndef(new_ptr_ty);
                }

                const offset_int = try sema.usizeCast(block, offset_src, offset_val.toUnsignedInt(target));
                // TODO I tried to put this check earlier but it the LLVM backend generate invalid instructinons
                if (offset_int == 0) return ptr;
                if (try ptr_val.getUnsignedIntAdvanced(target, sema.kit(block, ptr_src))) |addr| {
                    const ptr_child_ty = ptr_ty.childType();
                    const elem_ty = if (ptr_ty.isSinglePointer() and ptr_child_ty.zigTypeTag() == .Array)
                        ptr_child_ty.childType()
                    else
                        ptr_child_ty;

                    const elem_size = elem_ty.abiSize(target);
                    const new_addr = switch (air_tag) {
                        .ptr_add => addr + elem_size * offset_int,
                        .ptr_sub => addr - elem_size * offset_int,
                        else => unreachable,
                    };
                    const new_ptr_val = try Value.Tag.int_u64.create(sema.arena, new_addr);
                    return sema.addConstant(new_ptr_ty, new_ptr_val);
                }
                if (air_tag == .ptr_sub) {
                    return sema.fail(block, op_src, "TODO implement Sema comptime pointer subtraction", .{});
                }
                const new_ptr_val = try ptr_val.elemPtr(ptr_ty, sema.arena, offset_int, target);
                return sema.addConstant(new_ptr_ty, new_ptr_val);
            } else break :rs offset_src;
        } else break :rs ptr_src;
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addBinOp(air_tag, ptr, offset);
}

fn zirLoad(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ptr_src: LazySrcLoc = .{ .node_offset_deref_ptr = inst_data.src_node };
    const ptr = sema.resolveInst(inst_data.operand);
    return sema.analyzeLoad(block, src, ptr, ptr_src);
}

fn zirAsm(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const extra = sema.code.extraData(Zir.Inst.Asm, extended.operand);
    const src: LazySrcLoc = .{ .node_offset = extra.data.src_node };
    const ret_ty_src: LazySrcLoc = .{ .node_offset_asm_ret_ty = extra.data.src_node };
    const outputs_len = @truncate(u5, extended.small);
    const inputs_len = @truncate(u5, extended.small >> 5);
    const clobbers_len = @truncate(u5, extended.small >> 10);
    const is_volatile = @truncate(u1, extended.small >> 15) != 0;
    const is_global_assembly = sema.func == null;

    if (block.is_comptime and !is_global_assembly) {
        try sema.requireRuntimeBlock(block, src);
    }

    if (extra.data.asm_source == 0) {
        // This can move to become an AstGen error after inline assembly improvements land
        // and stage1 code matches stage2 code.
        return sema.fail(block, src, "assembly code must use string literal syntax", .{});
    }

    if (outputs_len > 1) {
        return sema.fail(block, src, "TODO implement Sema for asm with more than 1 output", .{});
    }

    var extra_i = extra.end;
    var output_type_bits = extra.data.output_type_bits;
    var needed_capacity: usize = @typeInfo(Air.Asm).Struct.fields.len + outputs_len + inputs_len;

    const Output = struct { constraint: []const u8, ty: Type };
    const output: ?Output = if (outputs_len == 0) null else blk: {
        const output = sema.code.extraData(Zir.Inst.Asm.Output, extra_i);
        extra_i = output.end;

        const is_type = @truncate(u1, output_type_bits) != 0;
        output_type_bits >>= 1;

        if (!is_type) {
            return sema.fail(block, src, "TODO implement Sema for asm with non `->` output", .{});
        }

        const constraint = sema.code.nullTerminatedString(output.data.constraint);
        needed_capacity += constraint.len / 4 + 1;

        break :blk Output{
            .constraint = constraint,
            .ty = try sema.resolveType(block, ret_ty_src, output.data.operand),
        };
    };

    const args = try sema.arena.alloc(Air.Inst.Ref, inputs_len);
    const inputs = try sema.arena.alloc(struct { c: []const u8, n: []const u8 }, inputs_len);

    for (args) |*arg, arg_i| {
        const input = sema.code.extraData(Zir.Inst.Asm.Input, extra_i);
        extra_i = input.end;

        const uncasted_arg = sema.resolveInst(input.data.operand);
        const uncasted_arg_ty = sema.typeOf(uncasted_arg);
        switch (uncasted_arg_ty.zigTypeTag()) {
            .ComptimeInt => arg.* = try sema.coerce(block, Type.initTag(.usize), uncasted_arg, src),
            .ComptimeFloat => arg.* = try sema.coerce(block, Type.initTag(.f64), uncasted_arg, src),
            else => arg.* = uncasted_arg,
        }

        const constraint = sema.code.nullTerminatedString(input.data.constraint);
        const name = sema.code.nullTerminatedString(input.data.name);
        needed_capacity += (constraint.len + name.len + 1) / 4 + 1;
        inputs[arg_i] = .{ .c = constraint, .n = name };
    }

    const clobbers = try sema.arena.alloc([]const u8, clobbers_len);
    for (clobbers) |*name| {
        name.* = sema.code.nullTerminatedString(sema.code.extra[extra_i]);
        extra_i += 1;

        needed_capacity += name.*.len / 4 + 1;
    }

    const asm_source = sema.code.nullTerminatedString(extra.data.asm_source);
    needed_capacity += (asm_source.len + 3) / 4;

    const gpa = sema.gpa;
    try sema.air_extra.ensureUnusedCapacity(gpa, needed_capacity);
    const asm_air = try block.addInst(.{
        .tag = .assembly,
        .data = .{ .ty_pl = .{
            .ty = if (output) |o| try sema.addType(o.ty) else Air.Inst.Ref.void_type,
            .payload = sema.addExtraAssumeCapacity(Air.Asm{
                .source_len = @intCast(u32, asm_source.len),
                .outputs_len = outputs_len,
                .inputs_len = @intCast(u32, args.len),
                .flags = (@as(u32, @boolToInt(is_volatile)) << 31) | @intCast(u32, clobbers.len),
            }),
        } },
    });
    if (output != null) {
        // Indicate the output is the asm instruction return value.
        sema.air_extra.appendAssumeCapacity(@enumToInt(Air.Inst.Ref.none));
    }
    sema.appendRefsAssumeCapacity(args);
    if (output) |o| {
        const buffer = mem.sliceAsBytes(sema.air_extra.unusedCapacitySlice());
        mem.copy(u8, buffer, o.constraint);
        buffer[o.constraint.len] = 0;
        sema.air_extra.items.len += o.constraint.len / 4 + 1;
    }
    for (inputs) |input| {
        const buffer = mem.sliceAsBytes(sema.air_extra.unusedCapacitySlice());
        mem.copy(u8, buffer, input.c);
        buffer[input.c.len] = 0;
        mem.copy(u8, buffer[input.c.len + 1 ..], input.n);
        buffer[input.c.len + 1 + input.n.len] = 0;
        sema.air_extra.items.len += (input.c.len + input.n.len + 1) / 4 + 1;
    }
    for (clobbers) |clobber| {
        const buffer = mem.sliceAsBytes(sema.air_extra.unusedCapacitySlice());
        mem.copy(u8, buffer, clobber);
        buffer[clobber.len] = 0;
        sema.air_extra.items.len += clobber.len / 4 + 1;
    }
    {
        const buffer = mem.sliceAsBytes(sema.air_extra.unusedCapacitySlice());
        mem.copy(u8, buffer, asm_source);
        sema.air_extra.items.len += (asm_source.len + 3) / 4;
    }
    return asm_air;
}

/// Only called for equality operators. See also `zirCmp`.
fn zirCmpEq(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    op: std.math.CompareOperator,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src: LazySrcLoc = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    const target = sema.mod.getTarget();

    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    const lhs_ty_tag = lhs_ty.zigTypeTag();
    const rhs_ty_tag = rhs_ty.zigTypeTag();
    if (lhs_ty_tag == .Null and rhs_ty_tag == .Null) {
        // null == null, null != null
        if (op == .eq) {
            return Air.Inst.Ref.bool_true;
        } else {
            return Air.Inst.Ref.bool_false;
        }
    }

    // comparing null with optionals
    if (lhs_ty_tag == .Null and (rhs_ty_tag == .Optional or rhs_ty.isCPtr())) {
        return sema.analyzeIsNull(block, src, rhs, op == .neq);
    }
    if (rhs_ty_tag == .Null and (lhs_ty_tag == .Optional or lhs_ty.isCPtr())) {
        return sema.analyzeIsNull(block, src, lhs, op == .neq);
    }

    if (lhs_ty_tag == .Null or rhs_ty_tag == .Null) {
        const non_null_type = if (lhs_ty_tag == .Null) rhs_ty else lhs_ty;
        return sema.fail(block, src, "comparison of '{}' with null", .{non_null_type.fmt(target)});
    }

    if (lhs_ty_tag == .Union and (rhs_ty_tag == .EnumLiteral or rhs_ty_tag == .Enum)) {
        return sema.analyzeCmpUnionTag(block, lhs, lhs_src, rhs, rhs_src, op);
    }
    if (rhs_ty_tag == .Union and (lhs_ty_tag == .EnumLiteral or lhs_ty_tag == .Enum)) {
        return sema.analyzeCmpUnionTag(block, rhs, rhs_src, lhs, lhs_src, op);
    }

    if (lhs_ty_tag == .ErrorSet and rhs_ty_tag == .ErrorSet) {
        const runtime_src: LazySrcLoc = src: {
            if (try sema.resolveMaybeUndefVal(block, lhs_src, lhs)) |lval| {
                if (try sema.resolveMaybeUndefVal(block, rhs_src, rhs)) |rval| {
                    if (lval.isUndef() or rval.isUndef()) {
                        return sema.addConstUndef(Type.bool);
                    }
                    // TODO optimisation opportunity: evaluate if mem.eql is faster with the names,
                    // or calling to Module.getErrorValue to get the values and then compare them is
                    // faster.
                    const lhs_name = lval.castTag(.@"error").?.data.name;
                    const rhs_name = rval.castTag(.@"error").?.data.name;
                    if (mem.eql(u8, lhs_name, rhs_name) == (op == .eq)) {
                        return Air.Inst.Ref.bool_true;
                    } else {
                        return Air.Inst.Ref.bool_false;
                    }
                } else {
                    break :src rhs_src;
                }
            } else {
                break :src lhs_src;
            }
        };
        try sema.requireRuntimeBlock(block, runtime_src);
        return block.addBinOp(air_tag, lhs, rhs);
    }
    if (lhs_ty_tag == .Type and rhs_ty_tag == .Type) {
        const lhs_as_type = try sema.analyzeAsType(block, lhs_src, lhs);
        const rhs_as_type = try sema.analyzeAsType(block, rhs_src, rhs);
        if (lhs_as_type.eql(rhs_as_type, target) == (op == .eq)) {
            return Air.Inst.Ref.bool_true;
        } else {
            return Air.Inst.Ref.bool_false;
        }
    }
    return sema.analyzeCmp(block, src, lhs, rhs, op, lhs_src, rhs_src, true);
}

fn analyzeCmpUnionTag(
    sema: *Sema,
    block: *Block,
    un: Air.Inst.Ref,
    un_src: LazySrcLoc,
    tag: Air.Inst.Ref,
    tag_src: LazySrcLoc,
    op: std.math.CompareOperator,
) CompileError!Air.Inst.Ref {
    const union_ty = try sema.resolveTypeFields(block, un_src, sema.typeOf(un));
    const union_tag_ty = union_ty.unionTagType() orelse {
        // TODO note at declaration site that says "union foo is not tagged"
        return sema.fail(block, un_src, "comparison of union and enum literal is only valid for tagged union types", .{});
    };
    // Coerce both the union and the tag to the union's tag type, and then execute the
    // enum comparison codepath.
    const coerced_tag = try sema.coerce(block, union_tag_ty, tag, tag_src);
    const coerced_union = try sema.coerce(block, union_tag_ty, un, un_src);

    return sema.cmpSelf(block, coerced_union, coerced_tag, op, un_src, tag_src);
}

/// Only called for non-equality operators. See also `zirCmpEq`.
fn zirCmp(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    op: std.math.CompareOperator,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src: LazySrcLoc = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    return sema.analyzeCmp(block, src, lhs, rhs, op, lhs_src, rhs_src, false);
}

fn analyzeCmp(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
    op: std.math.CompareOperator,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
    is_equality_cmp: bool,
) CompileError!Air.Inst.Ref {
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);

    if (lhs_ty.zigTypeTag() == .Vector and rhs_ty.zigTypeTag() == .Vector) {
        return sema.cmpVector(block, src, lhs, rhs, op, lhs_src, rhs_src);
    }
    if (lhs_ty.isNumeric() and rhs_ty.isNumeric()) {
        // This operation allows any combination of integer and float types, regardless of the
        // signed-ness, comptime-ness, and bit-width. So peer type resolution is incorrect for
        // numeric types.
        return sema.cmpNumeric(block, src, lhs, rhs, op, lhs_src, rhs_src);
    }
    const instructions = &[_]Air.Inst.Ref{ lhs, rhs };
    const resolved_type = try sema.resolvePeerTypes(block, src, instructions, .{ .override = &[_]LazySrcLoc{ lhs_src, rhs_src } });
    const target = sema.mod.getTarget();
    if (!resolved_type.isSelfComparable(is_equality_cmp)) {
        return sema.fail(block, src, "{s} operator not allowed for type '{}'", .{
            @tagName(op), resolved_type.fmt(target),
        });
    }
    const casted_lhs = try sema.coerce(block, resolved_type, lhs, lhs_src);
    const casted_rhs = try sema.coerce(block, resolved_type, rhs, rhs_src);
    return sema.cmpSelf(block, casted_lhs, casted_rhs, op, lhs_src, rhs_src);
}

fn cmpSelf(
    sema: *Sema,
    block: *Block,
    casted_lhs: Air.Inst.Ref,
    casted_rhs: Air.Inst.Ref,
    op: std.math.CompareOperator,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const resolved_type = sema.typeOf(casted_lhs);
    const target = sema.mod.getTarget();
    const runtime_src: LazySrcLoc = src: {
        if (try sema.resolveMaybeUndefVal(block, lhs_src, casted_lhs)) |lhs_val| {
            if (lhs_val.isUndef()) return sema.addConstUndef(Type.bool);
            if (try sema.resolveMaybeUndefVal(block, rhs_src, casted_rhs)) |rhs_val| {
                if (rhs_val.isUndef()) return sema.addConstUndef(Type.bool);

                if (resolved_type.zigTypeTag() == .Vector) {
                    const result_ty = try Type.vector(sema.arena, resolved_type.vectorLen(), Type.@"bool");
                    const cmp_val = try lhs_val.compareVector(op, rhs_val, resolved_type, sema.arena, target);
                    return sema.addConstant(result_ty, cmp_val);
                }

                if (lhs_val.compare(op, rhs_val, resolved_type, target)) {
                    return Air.Inst.Ref.bool_true;
                } else {
                    return Air.Inst.Ref.bool_false;
                }
            } else {
                if (resolved_type.zigTypeTag() == .Bool) {
                    // We can lower bool eq/neq more efficiently.
                    return sema.runtimeBoolCmp(block, op, casted_rhs, lhs_val.toBool(), rhs_src);
                }
                break :src rhs_src;
            }
        } else {
            // For bools, we still check the other operand, because we can lower
            // bool eq/neq more efficiently.
            if (resolved_type.zigTypeTag() == .Bool) {
                if (try sema.resolveMaybeUndefVal(block, rhs_src, casted_rhs)) |rhs_val| {
                    if (rhs_val.isUndef()) return sema.addConstUndef(Type.bool);
                    return sema.runtimeBoolCmp(block, op, casted_lhs, rhs_val.toBool(), lhs_src);
                }
            }
            break :src lhs_src;
        }
    };
    try sema.requireRuntimeBlock(block, runtime_src);
    if (resolved_type.zigTypeTag() == .Vector) {
        const result_ty = try Type.vector(sema.arena, resolved_type.vectorLen(), Type.@"bool");
        const result_ty_ref = try sema.addType(result_ty);
        return block.addCmpVector(casted_lhs, casted_rhs, op, result_ty_ref);
    }
    const tag = Air.Inst.Tag.fromCmpOp(op);
    return block.addBinOp(tag, casted_lhs, casted_rhs);
}

/// cmp_eq (x, false) => not(x)
/// cmp_eq (x, true ) => x
/// cmp_neq(x, false) => x
/// cmp_neq(x, true ) => not(x)
fn runtimeBoolCmp(
    sema: *Sema,
    block: *Block,
    op: std.math.CompareOperator,
    lhs: Air.Inst.Ref,
    rhs: bool,
    runtime_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    if ((op == .neq) == rhs) {
        try sema.requireRuntimeBlock(block, runtime_src);
        return block.addTyOp(.not, Type.bool, lhs);
    } else {
        return lhs;
    }
}

fn zirSizeOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_ty = try sema.resolveType(block, operand_src, inst_data.operand);
    try sema.resolveTypeLayout(block, src, operand_ty);
    const target = sema.mod.getTarget();
    const abi_size = switch (operand_ty.zigTypeTag()) {
        .Fn => unreachable,
        .NoReturn,
        .Undefined,
        .Null,
        .BoundFn,
        .Opaque,
        => return sema.fail(block, src, "no size available for type '{}'", .{operand_ty.fmt(target)}),

        .Type,
        .EnumLiteral,
        .ComptimeFloat,
        .ComptimeInt,
        .Void,
        => 0,

        .Bool,
        .Int,
        .Float,
        .Pointer,
        .Array,
        .Struct,
        .Optional,
        .ErrorUnion,
        .ErrorSet,
        .Enum,
        .Union,
        .Vector,
        .Frame,
        .AnyFrame,
        => operand_ty.abiSize(target),
    };
    return sema.addIntUnsigned(Type.comptime_int, abi_size);
}

fn zirBitSizeOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const unresolved_operand_ty = try sema.resolveType(block, operand_src, inst_data.operand);
    const operand_ty = try sema.resolveTypeFields(block, operand_src, unresolved_operand_ty);
    const target = sema.mod.getTarget();
    const bit_size = operand_ty.bitSize(target);
    return sema.addIntUnsigned(Type.initTag(.comptime_int), bit_size);
}

fn zirThis(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const this_decl = block.namespace.getDecl();
    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    return sema.analyzeDeclVal(block, src, this_decl);
}

fn zirClosureCapture(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!void {
    // TODO: Compile error when closed over values are modified
    const inst_data = sema.code.instructions.items(.data)[inst].un_tok;
    const src = inst_data.src();
    // Closures are not necessarily constant values. For example, the
    // code might do something like this:
    // fn foo(x: anytype) void { const S = struct {field: @TypeOf(x)}; }
    // ...in which case the closure_capture instruction has access to a runtime
    // value only. In such case we preserve the type and use a dummy runtime value.
    const operand = sema.resolveInst(inst_data.operand);
    const val = (try sema.resolveMaybeUndefValAllowVariables(block, src, operand)) orelse
        Value.initTag(.generic_poison);

    try block.wip_capture_scope.captures.putNoClobber(sema.gpa, inst, .{
        .ty = try sema.typeOf(operand).copy(sema.perm_arena),
        .val = try val.copy(sema.perm_arena),
    });
}

fn zirClosureGet(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    // TODO CLOSURE: Test this with inline functions
    const inst_data = sema.code.instructions.items(.data)[inst].inst_node;
    var scope: *CaptureScope = block.src_decl.src_scope.?;
    // Note: The target closure must be in this scope list.
    // If it's not here, the zir is invalid, or the list is broken.
    const tv = while (true) {
        // Note: We don't need to add a dependency here, because
        // decls always depend on their lexical parents.
        if (scope.captures.getPtr(inst_data.inst)) |tv| {
            break tv;
        }
        scope = scope.parent.?;
    } else unreachable;

    return sema.addConstant(tv.ty, tv.val);
}

fn zirRetAddr(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    try sema.requireRuntimeBlock(block, src);
    return try block.addNoOp(.ret_addr);
}

fn zirFrameAddress(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    try sema.requireRuntimeBlock(block, src);
    return try block.addNoOp(.frame_addr);
}

fn zirBuiltinSrc(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    const extra = sema.code.extraData(Zir.Inst.LineColumn, extended.operand).data;
    const func = sema.func orelse return sema.fail(block, src, "@src outside function", .{});

    const func_name_val = blk: {
        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();
        const name = std.mem.span(func.owner_decl.name);
        const bytes = try anon_decl.arena().dupe(u8, name[0 .. name.len + 1]);
        const new_decl = try anon_decl.finish(
            try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len - 1),
            try Value.Tag.bytes.create(anon_decl.arena(), bytes),
            0, // default alignment
        );
        break :blk try Value.Tag.decl_ref.create(sema.arena, new_decl);
    };

    const file_name_val = blk: {
        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();
        const name = try func.owner_decl.getFileScope().fullPathZ(anon_decl.arena());
        const new_decl = try anon_decl.finish(
            try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), name.len),
            try Value.Tag.bytes.create(anon_decl.arena(), name[0 .. name.len + 1]),
            0, // default alignment
        );
        break :blk try Value.Tag.decl_ref.create(sema.arena, new_decl);
    };

    const field_values = try sema.arena.alloc(Value, 4);
    // file: [:0]const u8,
    field_values[0] = file_name_val;
    // fn_name: [:0]const u8,
    field_values[1] = func_name_val;
    // line: u32
    field_values[2] = try Value.Tag.int_u64.create(sema.arena, extra.line + 1);
    // column: u32,
    field_values[3] = try Value.Tag.int_u64.create(sema.arena, extra.column + 1);

    return sema.addConstant(
        try sema.getBuiltinType(block, src, "SourceLocation"),
        try Value.Tag.aggregate.create(sema.arena, field_values),
    );
}

fn zirTypeInfo(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ty = try sema.resolveType(block, src, inst_data.operand);
    const type_info_ty = try sema.getBuiltinType(block, src, "Type");
    const target = sema.mod.getTarget();

    switch (ty.zigTypeTag()) {
        .Type => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Type)),
                .val = Value.@"void",
            }),
        ),
        .Void => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Void)),
                .val = Value.@"void",
            }),
        ),
        .Bool => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Bool)),
                .val = Value.@"void",
            }),
        ),
        .NoReturn => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.NoReturn)),
                .val = Value.@"void",
            }),
        ),
        .ComptimeFloat => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.ComptimeFloat)),
                .val = Value.@"void",
            }),
        ),
        .ComptimeInt => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.ComptimeInt)),
                .val = Value.@"void",
            }),
        ),
        .Undefined => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Undefined)),
                .val = Value.@"void",
            }),
        ),
        .Null => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Null)),
                .val = Value.@"void",
            }),
        ),
        .EnumLiteral => return sema.addConstant(
            type_info_ty,
            try Value.Tag.@"union".create(sema.arena, .{
                .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.EnumLiteral)),
                .val = Value.@"void",
            }),
        ),
        .Fn => {
            // TODO: look into memoizing this result.
            const info = ty.fnInfo();

            var params_anon_decl = try block.startAnonDecl(src);
            defer params_anon_decl.deinit();

            const param_vals = try params_anon_decl.arena().alloc(Value, info.param_types.len);
            for (param_vals) |*param_val, i| {
                const param_ty = info.param_types[i];
                const is_generic = param_ty.tag() == .generic_poison;
                const param_ty_val = if (is_generic)
                    Value.@"null"
                else
                    try Value.Tag.opt_payload.create(
                        params_anon_decl.arena(),
                        try Value.Tag.ty.create(params_anon_decl.arena(), param_ty),
                    );

                const param_fields = try params_anon_decl.arena().create([3]Value);
                param_fields.* = .{
                    // is_generic: bool,
                    Value.makeBool(is_generic),
                    // is_noalias: bool,
                    Value.@"false", // TODO
                    // arg_type: ?type,
                    param_ty_val,
                };
                param_val.* = try Value.Tag.aggregate.create(params_anon_decl.arena(), param_fields);
            }

            const args_val = v: {
                const fn_info_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    type_info_ty.getNamespace().?,
                    "Fn",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, fn_info_decl);
                try sema.ensureDeclAnalyzed(fn_info_decl);
                var fn_ty_buffer: Value.ToTypeBuffer = undefined;
                const fn_ty = fn_info_decl.val.toType(&fn_ty_buffer);
                const param_info_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    fn_ty.getNamespace().?,
                    "Param",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, param_info_decl);
                try sema.ensureDeclAnalyzed(param_info_decl);
                var param_buffer: Value.ToTypeBuffer = undefined;
                const param_ty = param_info_decl.val.toType(&param_buffer);
                const new_decl = try params_anon_decl.finish(
                    try Type.Tag.array.create(params_anon_decl.arena(), .{
                        .len = param_vals.len,
                        .elem_type = try param_ty.copy(params_anon_decl.arena()),
                    }),
                    try Value.Tag.aggregate.create(
                        params_anon_decl.arena(),
                        param_vals,
                    ),
                    0, // default alignment
                );
                break :v try Value.Tag.slice.create(sema.arena, .{
                    .ptr = try Value.Tag.decl_ref.create(sema.arena, new_decl),
                    .len = try Value.Tag.int_u64.create(sema.arena, param_vals.len),
                });
            };

            const ret_ty_opt = if (info.return_type.tag() != .generic_poison)
                try Value.Tag.opt_payload.create(
                    sema.arena,
                    try Value.Tag.ty.create(sema.arena, info.return_type),
                )
            else
                Value.@"null";

            const field_values = try sema.arena.create([6]Value);
            field_values.* = .{
                // calling_convention: CallingConvention,
                try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(info.cc)),
                // alignment: comptime_int,
                try Value.Tag.int_u64.create(sema.arena, ty.abiAlignment(target)),
                // is_generic: bool,
                Value.makeBool(info.is_generic),
                // is_var_args: bool,
                Value.makeBool(info.is_var_args),
                // return_type: ?type,
                ret_ty_opt,
                // args: []const Fn.Param,
                args_val,
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Fn)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Int => {
            const info = ty.intInfo(target);
            const field_values = try sema.arena.alloc(Value, 2);
            // signedness: Signedness,
            field_values[0] = try Value.Tag.enum_field_index.create(
                sema.arena,
                @enumToInt(info.signedness),
            );
            // bits: comptime_int,
            field_values[1] = try Value.Tag.int_u64.create(sema.arena, info.bits);

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Int)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Float => {
            const field_values = try sema.arena.alloc(Value, 1);
            // bits: comptime_int,
            field_values[0] = try Value.Tag.int_u64.create(sema.arena, ty.bitSize(target));

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Float)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Pointer => {
            const info = ty.ptrInfo().data;
            const alignment = if (info.@"align" != 0)
                try Value.Tag.int_u64.create(sema.arena, info.@"align")
            else
                try info.pointee_type.lazyAbiAlignment(target, sema.arena);

            const field_values = try sema.arena.create([8]Value);
            field_values.* = .{
                // size: Size,
                try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(info.size)),
                // is_const: bool,
                Value.makeBool(!info.mutable),
                // is_volatile: bool,
                Value.makeBool(info.@"volatile"),
                // alignment: comptime_int,
                alignment,
                // address_space: AddressSpace
                try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(info.@"addrspace")),
                // child: type,
                try Value.Tag.ty.create(sema.arena, info.pointee_type),
                // is_allowzero: bool,
                Value.makeBool(info.@"allowzero"),
                // sentinel: ?*const anyopaque,
                try sema.optRefValue(block, src, info.pointee_type, info.sentinel),
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Pointer)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Array => {
            const info = ty.arrayInfo();
            const field_values = try sema.arena.alloc(Value, 3);
            // len: comptime_int,
            field_values[0] = try Value.Tag.int_u64.create(sema.arena, info.len);
            // child: type,
            field_values[1] = try Value.Tag.ty.create(sema.arena, info.elem_type);
            // sentinel: ?*const anyopaque,
            field_values[2] = try sema.optRefValue(block, src, info.elem_type, info.sentinel);

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Array)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Vector => {
            const info = ty.arrayInfo();
            const field_values = try sema.arena.alloc(Value, 2);
            // len: comptime_int,
            field_values[0] = try Value.Tag.int_u64.create(sema.arena, info.len);
            // child: type,
            field_values[1] = try Value.Tag.ty.create(sema.arena, info.elem_type);

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Vector)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Optional => {
            const field_values = try sema.arena.alloc(Value, 1);
            // child: type,
            field_values[0] = try Value.Tag.ty.create(sema.arena, try ty.optionalChildAlloc(sema.arena));

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Optional)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .ErrorSet => {
            var fields_anon_decl = try block.startAnonDecl(src);
            defer fields_anon_decl.deinit();

            // Get the Error type
            const error_field_ty = t: {
                const set_field_ty_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    type_info_ty.getNamespace().?,
                    "Error",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, set_field_ty_decl);
                try sema.ensureDeclAnalyzed(set_field_ty_decl);
                var buffer: Value.ToTypeBuffer = undefined;
                break :t try set_field_ty_decl.val.toType(&buffer).copy(fields_anon_decl.arena());
            };

            try sema.queueFullTypeResolution(try error_field_ty.copy(sema.arena));

            // If the error set is inferred it has to be resolved at this point
            try sema.resolveInferredErrorSetTy(block, src, ty);

            // Build our list of Error values
            // Optional value is only null if anyerror
            // Value can be zero-length slice otherwise
            const error_field_vals: ?[]Value = if (ty.isAnyError()) null else blk: {
                const names = ty.errorSetNames();
                const vals = try fields_anon_decl.arena().alloc(Value, names.len);
                for (vals) |*field_val, i| {
                    const name = names[i];
                    const name_val = v: {
                        var anon_decl = try block.startAnonDecl(src);
                        defer anon_decl.deinit();
                        const bytes = try anon_decl.arena().dupeZ(u8, name);
                        const new_decl = try anon_decl.finish(
                            try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                            try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                            0, // default alignment
                        );
                        break :v try Value.Tag.decl_ref.create(fields_anon_decl.arena(), new_decl);
                    };

                    const error_field_fields = try fields_anon_decl.arena().create([1]Value);
                    error_field_fields.* = .{
                        // name: []const u8,
                        name_val,
                    };

                    field_val.* = try Value.Tag.aggregate.create(
                        fields_anon_decl.arena(),
                        error_field_fields,
                    );
                }

                break :blk vals;
            };

            // Build our ?[]const Error value
            const errors_val = if (error_field_vals) |vals| v: {
                const new_decl = try fields_anon_decl.finish(
                    try Type.Tag.array.create(fields_anon_decl.arena(), .{
                        .len = vals.len,
                        .elem_type = error_field_ty,
                    }),
                    try Value.Tag.aggregate.create(
                        fields_anon_decl.arena(),
                        vals,
                    ),
                    0, // default alignment
                );

                const new_decl_val = try Value.Tag.decl_ref.create(sema.arena, new_decl);
                const slice_val = try Value.Tag.slice.create(sema.arena, .{
                    .ptr = new_decl_val,
                    .len = try Value.Tag.int_u64.create(sema.arena, vals.len),
                });
                break :v try Value.Tag.opt_payload.create(sema.arena, slice_val);
            } else Value.@"null";

            // Construct Type{ .ErrorSet = errors_val }
            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.ErrorSet)),
                    .val = errors_val,
                }),
            );
        },
        .ErrorUnion => {
            const field_values = try sema.arena.alloc(Value, 2);
            // error_set: type,
            field_values[0] = try Value.Tag.ty.create(sema.arena, ty.errorUnionSet());
            // payload: type,
            field_values[1] = try Value.Tag.ty.create(sema.arena, ty.errorUnionPayload());

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.ErrorUnion)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Enum => {
            // TODO: look into memoizing this result.
            var int_tag_type_buffer: Type.Payload.Bits = undefined;
            const int_tag_ty = try ty.intTagType(&int_tag_type_buffer).copy(sema.arena);

            const is_exhaustive = Value.makeBool(!ty.isNonexhaustiveEnum());

            var fields_anon_decl = try block.startAnonDecl(src);
            defer fields_anon_decl.deinit();

            const enum_field_ty = t: {
                const enum_field_ty_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    type_info_ty.getNamespace().?,
                    "EnumField",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, enum_field_ty_decl);
                try sema.ensureDeclAnalyzed(enum_field_ty_decl);
                var buffer: Value.ToTypeBuffer = undefined;
                break :t try enum_field_ty_decl.val.toType(&buffer).copy(fields_anon_decl.arena());
            };

            const enum_fields = ty.enumFields();
            const enum_field_vals = try fields_anon_decl.arena().alloc(Value, enum_fields.count());

            for (enum_field_vals) |*field_val, i| {
                var tag_val_payload: Value.Payload.U32 = .{
                    .base = .{ .tag = .enum_field_index },
                    .data = @intCast(u32, i),
                };
                const tag_val = Value.initPayload(&tag_val_payload.base);

                var buffer: Value.Payload.U64 = undefined;
                const int_val = try tag_val.enumToInt(ty, &buffer).copy(fields_anon_decl.arena());

                const name = enum_fields.keys()[i];
                const name_val = v: {
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    const bytes = try anon_decl.arena().dupeZ(u8, name);
                    const new_decl = try anon_decl.finish(
                        try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                        try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                        0, // default alignment
                    );
                    break :v try Value.Tag.decl_ref.create(fields_anon_decl.arena(), new_decl);
                };

                const enum_field_fields = try fields_anon_decl.arena().create([2]Value);
                enum_field_fields.* = .{
                    // name: []const u8,
                    name_val,
                    // value: comptime_int,
                    int_val,
                };
                field_val.* = try Value.Tag.aggregate.create(fields_anon_decl.arena(), enum_field_fields);
            }

            const fields_val = v: {
                const new_decl = try fields_anon_decl.finish(
                    try Type.Tag.array.create(fields_anon_decl.arena(), .{
                        .len = enum_field_vals.len,
                        .elem_type = enum_field_ty,
                    }),
                    try Value.Tag.aggregate.create(
                        fields_anon_decl.arena(),
                        enum_field_vals,
                    ),
                    0, // default alignment
                );
                break :v try Value.Tag.decl_ref.create(sema.arena, new_decl);
            };

            const decls_val = try sema.typeInfoDecls(block, src, type_info_ty, ty.getNamespace());

            const field_values = try sema.arena.create([5]Value);
            field_values.* = .{
                // layout: ContainerLayout,
                try Value.Tag.enum_field_index.create(
                    sema.arena,
                    @enumToInt(std.builtin.Type.ContainerLayout.Auto),
                ),

                // tag_type: type,
                try Value.Tag.ty.create(sema.arena, int_tag_ty),
                // fields: []const EnumField,
                fields_val,
                // decls: []const Declaration,
                decls_val,
                // is_exhaustive: bool,
                is_exhaustive,
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Enum)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Union => {
            // TODO: look into memoizing this result.

            var fields_anon_decl = try block.startAnonDecl(src);
            defer fields_anon_decl.deinit();

            const union_field_ty = t: {
                const union_field_ty_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    type_info_ty.getNamespace().?,
                    "UnionField",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, union_field_ty_decl);
                try sema.ensureDeclAnalyzed(union_field_ty_decl);
                var buffer: Value.ToTypeBuffer = undefined;
                break :t try union_field_ty_decl.val.toType(&buffer).copy(fields_anon_decl.arena());
            };

            const union_ty = try sema.resolveTypeFields(block, src, ty);
            try sema.resolveTypeLayout(block, src, ty); // Getting alignment requires type layout
            const layout = union_ty.containerLayout();

            const union_fields = union_ty.unionFields();
            const union_field_vals = try fields_anon_decl.arena().alloc(Value, union_fields.count());

            for (union_field_vals) |*field_val, i| {
                const field = union_fields.values()[i];
                const name = union_fields.keys()[i];
                const name_val = v: {
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    const bytes = try anon_decl.arena().dupeZ(u8, name);
                    const new_decl = try anon_decl.finish(
                        try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                        try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                        0, // default alignment
                    );
                    break :v try Value.Tag.decl_ref.create(fields_anon_decl.arena(), new_decl);
                };

                const union_field_fields = try fields_anon_decl.arena().create([3]Value);
                const alignment = switch (layout) {
                    .Auto, .Extern => try sema.unionFieldAlignment(block, src, field),
                    .Packed => 0,
                };

                union_field_fields.* = .{
                    // name: []const u8,
                    name_val,
                    // field_type: type,
                    try Value.Tag.ty.create(fields_anon_decl.arena(), field.ty),
                    // alignment: comptime_int,
                    try Value.Tag.int_u64.create(fields_anon_decl.arena(), alignment),
                };
                field_val.* = try Value.Tag.aggregate.create(fields_anon_decl.arena(), union_field_fields);
            }

            const fields_val = v: {
                const new_decl = try fields_anon_decl.finish(
                    try Type.Tag.array.create(fields_anon_decl.arena(), .{
                        .len = union_field_vals.len,
                        .elem_type = union_field_ty,
                    }),
                    try Value.Tag.aggregate.create(
                        fields_anon_decl.arena(),
                        try fields_anon_decl.arena().dupe(Value, union_field_vals),
                    ),
                    0, // default alignment
                );
                break :v try Value.Tag.slice.create(sema.arena, .{
                    .ptr = try Value.Tag.decl_ref.create(sema.arena, new_decl),
                    .len = try Value.Tag.int_u64.create(sema.arena, union_field_vals.len),
                });
            };

            const decls_val = try sema.typeInfoDecls(block, src, type_info_ty, union_ty.getNamespace());

            const enum_tag_ty_val = if (union_ty.unionTagType()) |tag_ty| v: {
                const ty_val = try Value.Tag.ty.create(sema.arena, tag_ty);
                break :v try Value.Tag.opt_payload.create(sema.arena, ty_val);
            } else Value.@"null";

            const field_values = try sema.arena.create([4]Value);
            field_values.* = .{
                // layout: ContainerLayout,
                try Value.Tag.enum_field_index.create(
                    sema.arena,
                    @enumToInt(layout),
                ),

                // tag_type: ?type,
                enum_tag_ty_val,
                // fields: []const UnionField,
                fields_val,
                // decls: []const Declaration,
                decls_val,
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Union)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Struct => {
            // TODO: look into memoizing this result.

            var fields_anon_decl = try block.startAnonDecl(src);
            defer fields_anon_decl.deinit();

            const struct_field_ty = t: {
                const struct_field_ty_decl = (try sema.namespaceLookup(
                    block,
                    src,
                    type_info_ty.getNamespace().?,
                    "StructField",
                )).?;
                try sema.mod.declareDeclDependency(sema.owner_decl, struct_field_ty_decl);
                try sema.ensureDeclAnalyzed(struct_field_ty_decl);
                var buffer: Value.ToTypeBuffer = undefined;
                break :t try struct_field_ty_decl.val.toType(&buffer).copy(fields_anon_decl.arena());
            };
            const struct_ty = try sema.resolveTypeFields(block, src, ty);
            try sema.resolveTypeLayout(block, src, ty); // Getting alignment requires type layout
            const layout = struct_ty.containerLayout();

            const struct_field_vals = fv: {
                if (struct_ty.isTupleOrAnonStruct()) {
                    const tuple = struct_ty.tupleFields();
                    const field_types = tuple.types;
                    const struct_field_vals = try fields_anon_decl.arena().alloc(Value, field_types.len);
                    for (struct_field_vals) |*struct_field_val, i| {
                        const field_ty = field_types[i];
                        const name_val = v: {
                            var anon_decl = try block.startAnonDecl(src);
                            defer anon_decl.deinit();
                            const bytes = if (struct_ty.castTag(.anon_struct)) |payload|
                                try anon_decl.arena().dupeZ(u8, payload.data.names[i])
                            else
                                try std.fmt.allocPrintZ(anon_decl.arena(), "{d}", .{i});
                            const new_decl = try anon_decl.finish(
                                try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                                try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                                0, // default alignment
                            );
                            break :v try Value.Tag.slice.create(fields_anon_decl.arena(), .{
                                .ptr = try Value.Tag.decl_ref.create(fields_anon_decl.arena(), new_decl),
                                .len = try Value.Tag.int_u64.create(fields_anon_decl.arena(), bytes.len),
                            });
                        };

                        const struct_field_fields = try fields_anon_decl.arena().create([5]Value);
                        const field_val = tuple.values[i];
                        const is_comptime = field_val.tag() != .unreachable_value;
                        const opt_default_val = if (is_comptime) field_val else null;
                        const default_val_ptr = try sema.optRefValue(block, src, field_ty, opt_default_val);
                        struct_field_fields.* = .{
                            // name: []const u8,
                            name_val,
                            // field_type: type,
                            try Value.Tag.ty.create(fields_anon_decl.arena(), field_ty),
                            // default_value: ?*const anyopaque,
                            try default_val_ptr.copy(fields_anon_decl.arena()),
                            // is_comptime: bool,
                            Value.makeBool(is_comptime),
                            // alignment: comptime_int,
                            try field_ty.lazyAbiAlignment(target, fields_anon_decl.arena()),
                        };
                        struct_field_val.* = try Value.Tag.aggregate.create(fields_anon_decl.arena(), struct_field_fields);
                    }
                    break :fv struct_field_vals;
                }
                const struct_fields = struct_ty.structFields();
                const struct_field_vals = try fields_anon_decl.arena().alloc(Value, struct_fields.count());

                for (struct_field_vals) |*field_val, i| {
                    const field = struct_fields.values()[i];
                    const name = struct_fields.keys()[i];
                    const name_val = v: {
                        var anon_decl = try block.startAnonDecl(src);
                        defer anon_decl.deinit();
                        const bytes = try anon_decl.arena().dupeZ(u8, name);
                        const new_decl = try anon_decl.finish(
                            try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                            try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                            0, // default alignment
                        );
                        break :v try Value.Tag.slice.create(fields_anon_decl.arena(), .{
                            .ptr = try Value.Tag.decl_ref.create(fields_anon_decl.arena(), new_decl),
                            .len = try Value.Tag.int_u64.create(fields_anon_decl.arena(), bytes.len),
                        });
                    };

                    const struct_field_fields = try fields_anon_decl.arena().create([5]Value);
                    const opt_default_val = if (field.default_val.tag() == .unreachable_value)
                        null
                    else
                        field.default_val;
                    const default_val_ptr = try sema.optRefValue(block, src, field.ty, opt_default_val);
                    const alignment = switch (layout) {
                        .Auto, .Extern => field.normalAlignment(target),
                        .Packed => 0,
                    };

                    struct_field_fields.* = .{
                        // name: []const u8,
                        name_val,
                        // field_type: type,
                        try Value.Tag.ty.create(fields_anon_decl.arena(), field.ty),
                        // default_value: ?*const anyopaque,
                        try default_val_ptr.copy(fields_anon_decl.arena()),
                        // is_comptime: bool,
                        Value.makeBool(field.is_comptime),
                        // alignment: comptime_int,
                        try Value.Tag.int_u64.create(fields_anon_decl.arena(), alignment),
                    };
                    field_val.* = try Value.Tag.aggregate.create(fields_anon_decl.arena(), struct_field_fields);
                }
                break :fv struct_field_vals;
            };

            const fields_val = v: {
                const new_decl = try fields_anon_decl.finish(
                    try Type.Tag.array.create(fields_anon_decl.arena(), .{
                        .len = struct_field_vals.len,
                        .elem_type = struct_field_ty,
                    }),
                    try Value.Tag.aggregate.create(
                        fields_anon_decl.arena(),
                        try fields_anon_decl.arena().dupe(Value, struct_field_vals),
                    ),
                    0, // default alignment
                );
                break :v try Value.Tag.slice.create(sema.arena, .{
                    .ptr = try Value.Tag.decl_ref.create(sema.arena, new_decl),
                    .len = try Value.Tag.int_u64.create(sema.arena, struct_field_vals.len),
                });
            };

            const decls_val = try sema.typeInfoDecls(block, src, type_info_ty, struct_ty.getNamespace());

            const field_values = try sema.arena.create([4]Value);
            field_values.* = .{
                // layout: ContainerLayout,
                try Value.Tag.enum_field_index.create(
                    sema.arena,
                    @enumToInt(layout),
                ),
                // fields: []const StructField,
                fields_val,
                // decls: []const Declaration,
                decls_val,
                // is_tuple: bool,
                Value.makeBool(struct_ty.isTuple()),
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Struct)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .Opaque => {
            // TODO: look into memoizing this result.

            const opaque_ty = try sema.resolveTypeFields(block, src, ty);
            const decls_val = try sema.typeInfoDecls(block, src, type_info_ty, opaque_ty.getNamespace());

            const field_values = try sema.arena.create([1]Value);
            field_values.* = .{
                // decls: []const Declaration,
                decls_val,
            };

            return sema.addConstant(
                type_info_ty,
                try Value.Tag.@"union".create(sema.arena, .{
                    .tag = try Value.Tag.enum_field_index.create(sema.arena, @enumToInt(std.builtin.TypeId.Opaque)),
                    .val = try Value.Tag.aggregate.create(sema.arena, field_values),
                }),
            );
        },
        .BoundFn => @panic("TODO remove this type from the language and compiler"),
        .Frame => return sema.fail(block, src, "TODO: implement zirTypeInfo for Frame", .{}),
        .AnyFrame => return sema.fail(block, src, "TODO: implement zirTypeInfo for AnyFrame", .{}),
    }
}

fn typeInfoDecls(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    type_info_ty: Type,
    opt_namespace: ?*Module.Namespace,
) CompileError!Value {
    var decls_anon_decl = try block.startAnonDecl(src);
    defer decls_anon_decl.deinit();

    const declaration_ty = t: {
        const declaration_ty_decl = (try sema.namespaceLookup(
            block,
            src,
            type_info_ty.getNamespace().?,
            "Declaration",
        )).?;
        try sema.mod.declareDeclDependency(sema.owner_decl, declaration_ty_decl);
        try sema.ensureDeclAnalyzed(declaration_ty_decl);
        var buffer: Value.ToTypeBuffer = undefined;
        break :t try declaration_ty_decl.val.toType(&buffer).copy(decls_anon_decl.arena());
    };
    try sema.queueFullTypeResolution(try declaration_ty.copy(sema.arena));

    const decls_len = if (opt_namespace) |ns| ns.decls.count() else 0;
    const decls_vals = try decls_anon_decl.arena().alloc(Value, decls_len);
    for (decls_vals) |*decls_val, i| {
        const decl = opt_namespace.?.decls.keys()[i];
        const name_val = v: {
            var anon_decl = try block.startAnonDecl(src);
            defer anon_decl.deinit();
            const bytes = try anon_decl.arena().dupeZ(u8, mem.sliceTo(decl.name, 0));
            const new_decl = try anon_decl.finish(
                try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
                try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
                0, // default alignment
            );
            break :v try Value.Tag.slice.create(decls_anon_decl.arena(), .{
                .ptr = try Value.Tag.decl_ref.create(decls_anon_decl.arena(), new_decl),
                .len = try Value.Tag.int_u64.create(decls_anon_decl.arena(), bytes.len),
            });
        };

        const fields = try decls_anon_decl.arena().create([2]Value);
        fields.* = .{
            //name: []const u8,
            name_val,
            //is_pub: bool,
            Value.makeBool(decl.is_pub),
        };
        decls_val.* = try Value.Tag.aggregate.create(decls_anon_decl.arena(), fields);
    }

    const new_decl = try decls_anon_decl.finish(
        try Type.Tag.array.create(decls_anon_decl.arena(), .{
            .len = decls_vals.len,
            .elem_type = declaration_ty,
        }),
        try Value.Tag.aggregate.create(
            decls_anon_decl.arena(),
            try decls_anon_decl.arena().dupe(Value, decls_vals),
        ),
        0, // default alignment
    );
    return try Value.Tag.slice.create(sema.arena, .{
        .ptr = try Value.Tag.decl_ref.create(sema.arena, new_decl),
        .len = try Value.Tag.int_u64.create(sema.arena, decls_vals.len),
    });
}

fn zirTypeof(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    _ = block;
    const zir_datas = sema.code.instructions.items(.data);
    const inst_data = zir_datas[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    return sema.addType(operand_ty);
}

fn zirTypeofBuiltin(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const pl_node = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Block, pl_node.payload_index);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];

    var child_block: Block = .{
        .parent = block,
        .sema = sema,
        .src_decl = block.src_decl,
        .namespace = block.namespace,
        .wip_capture_scope = block.wip_capture_scope,
        .instructions = .{},
        .inlining = block.inlining,
        .is_comptime = false,
        .is_typeof = true,
    };
    defer child_block.instructions.deinit(sema.gpa);

    const operand = try sema.resolveBody(&child_block, body, inst);
    const operand_ty = sema.typeOf(operand);
    if (operand_ty.tag() == .generic_poison) return error.GenericPoison;
    return sema.addType(operand_ty);
}

fn zirTypeofLog2IntType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const res_ty = try sema.log2IntType(block, operand_ty, src);
    return sema.addType(res_ty);
}

fn zirLog2IntType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = try sema.resolveType(block, src, inst_data.operand);
    const res_ty = try sema.log2IntType(block, operand, src);
    return sema.addType(res_ty);
}

fn log2IntType(sema: *Sema, block: *Block, operand: Type, src: LazySrcLoc) CompileError!Type {
    switch (operand.zigTypeTag()) {
        .ComptimeInt => return Type.@"comptime_int",
        .Int => {
            const bits = operand.bitSize(sema.mod.getTarget());
            const count = if (bits == 0)
                0
            else blk: {
                var count: u16 = 0;
                var s = bits - 1;
                while (s != 0) : (s >>= 1) {
                    count += 1;
                }
                break :blk count;
            };
            return Module.makeIntType(sema.arena, .unsigned, count);
        },
        .Vector => {
            const elem_ty = operand.elemType2();
            const log2_elem_ty = try sema.log2IntType(block, elem_ty, src);
            return Type.Tag.vector.create(sema.arena, .{
                .len = operand.vectorLen(),
                .elem_type = log2_elem_ty,
            });
        },
        else => {},
    }
    const target = sema.mod.getTarget();
    return sema.fail(
        block,
        src,
        "bit shifting operation expected integer type, found '{}'",
        .{operand.fmt(target)},
    );
}

fn zirTypeofPeer(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const extra = sema.code.extraData(Zir.Inst.TypeOfPeer, extended.operand);
    const src: LazySrcLoc = .{ .node_offset = extra.data.src_node };
    const body = sema.code.extra[extra.data.body_index..][0..extra.data.body_len];

    var child_block: Block = .{
        .parent = block,
        .sema = sema,
        .src_decl = block.src_decl,
        .namespace = block.namespace,
        .wip_capture_scope = block.wip_capture_scope,
        .instructions = .{},
        .inlining = block.inlining,
        .is_comptime = false,
        .is_typeof = true,
    };
    defer child_block.instructions.deinit(sema.gpa);
    // Ignore the result, we only care about the instructions in `args`.
    _ = try sema.analyzeBodyBreak(&child_block, body);

    const args = sema.code.refSlice(extra.end, extended.small);

    const inst_list = try sema.gpa.alloc(Air.Inst.Ref, args.len);
    defer sema.gpa.free(inst_list);

    for (args) |arg_ref, i| {
        inst_list[i] = sema.resolveInst(arg_ref);
        if (sema.typeOf(inst_list[i]).tag() == .generic_poison) return error.GenericPoison;
    }

    const result_type = try sema.resolvePeerTypes(block, src, inst_list, .{ .typeof_builtin_call_node_offset = extra.data.src_node });
    return sema.addType(result_type);
}

fn zirBoolNot(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand_src = src; // TODO put this on the operand, not the `!`
    const uncasted_operand = sema.resolveInst(inst_data.operand);

    const operand = try sema.coerce(block, Type.bool, uncasted_operand, operand_src);
    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        return if (val.isUndef())
            sema.addConstUndef(Type.bool)
        else if (val.toBool())
            Air.Inst.Ref.bool_false
        else
            Air.Inst.Ref.bool_true;
    }
    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.not, Type.bool, operand);
}

fn zirBoolBr(
    sema: *Sema,
    parent_block: *Block,
    inst: Zir.Inst.Index,
    is_bool_or: bool,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const datas = sema.code.instructions.items(.data);
    const inst_data = datas[inst].bool_br;
    const lhs = sema.resolveInst(inst_data.lhs);
    const lhs_src = sema.src;
    const extra = sema.code.extraData(Zir.Inst.Block, inst_data.payload_index);
    const body = sema.code.extra[extra.end..][0..extra.data.body_len];
    const gpa = sema.gpa;

    if (try sema.resolveDefinedValue(parent_block, lhs_src, lhs)) |lhs_val| {
        if (lhs_val.toBool() == is_bool_or) {
            if (is_bool_or) {
                return Air.Inst.Ref.bool_true;
            } else {
                return Air.Inst.Ref.bool_false;
            }
        }
        // comptime-known left-hand side. No need for a block here; the result
        // is simply the rhs expression. Here we rely on there only being 1
        // break instruction (`break_inline`).
        return sema.resolveBody(parent_block, body, inst);
    }

    const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
    try sema.air_instructions.append(gpa, .{
        .tag = .block,
        .data = .{ .ty_pl = .{
            .ty = .bool_type,
            .payload = undefined,
        } },
    });

    var child_block = parent_block.makeSubBlock();
    child_block.runtime_loop = null;
    child_block.runtime_cond = lhs_src;
    child_block.runtime_index += 1;
    defer child_block.instructions.deinit(gpa);

    var then_block = child_block.makeSubBlock();
    defer then_block.instructions.deinit(gpa);

    var else_block = child_block.makeSubBlock();
    defer else_block.instructions.deinit(gpa);

    const lhs_block = if (is_bool_or) &then_block else &else_block;
    const rhs_block = if (is_bool_or) &else_block else &then_block;

    const lhs_result: Air.Inst.Ref = if (is_bool_or) .bool_true else .bool_false;
    _ = try lhs_block.addBr(block_inst, lhs_result);

    const rhs_result = try sema.resolveBody(rhs_block, body, inst);
    _ = try rhs_block.addBr(block_inst, rhs_result);

    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.CondBr).Struct.fields.len +
        then_block.instructions.items.len + else_block.instructions.items.len +
        @typeInfo(Air.Block).Struct.fields.len + child_block.instructions.items.len + 1);

    const cond_br_payload = sema.addExtraAssumeCapacity(Air.CondBr{
        .then_body_len = @intCast(u32, then_block.instructions.items.len),
        .else_body_len = @intCast(u32, else_block.instructions.items.len),
    });
    sema.air_extra.appendSliceAssumeCapacity(then_block.instructions.items);
    sema.air_extra.appendSliceAssumeCapacity(else_block.instructions.items);

    _ = try child_block.addInst(.{ .tag = .cond_br, .data = .{ .pl_op = .{
        .operand = lhs,
        .payload = cond_br_payload,
    } } });

    sema.air_instructions.items(.data)[block_inst].ty_pl.payload = sema.addExtraAssumeCapacity(
        Air.Block{ .body_len = @intCast(u32, child_block.instructions.items.len) },
    );
    sema.air_extra.appendSliceAssumeCapacity(child_block.instructions.items);

    try parent_block.instructions.append(gpa, block_inst);
    return Air.indexToRef(block_inst);
}

fn zirIsNonNull(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    return sema.analyzeIsNull(block, src, operand, true);
}

fn zirIsNonNullPtr(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ptr = sema.resolveInst(inst_data.operand);
    if ((try sema.resolveMaybeUndefVal(block, src, ptr)) == null) {
        return block.addUnOp(.is_non_null_ptr, ptr);
    }
    const loaded = try sema.analyzeLoad(block, src, ptr, src);
    return sema.analyzeIsNull(block, src, loaded, true);
}

fn zirIsNonErr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    return sema.analyzeIsNonErr(block, inst_data.src(), operand);
}

fn zirIsNonErrPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ptr = sema.resolveInst(inst_data.operand);
    const loaded = try sema.analyzeLoad(block, src, ptr, src);
    return sema.analyzeIsNonErr(block, src, loaded);
}

fn zirCondbr(
    sema: *Sema,
    parent_block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const cond_src: LazySrcLoc = .{ .node_offset_if_cond = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.CondBr, inst_data.payload_index);

    const then_body = sema.code.extra[extra.end..][0..extra.data.then_body_len];
    const else_body = sema.code.extra[extra.end + then_body.len ..][0..extra.data.else_body_len];

    const uncasted_cond = sema.resolveInst(extra.data.condition);
    const cond = try sema.coerce(parent_block, Type.bool, uncasted_cond, cond_src);

    if (try sema.resolveDefinedValue(parent_block, src, cond)) |cond_val| {
        const body = if (cond_val.toBool()) then_body else else_body;
        // We use `analyzeBodyInner` since we want to propagate any possible
        // `error.ComptimeBreak` to the caller.
        return sema.analyzeBodyInner(parent_block, body);
    }

    const gpa = sema.gpa;

    // We'll re-use the sub block to save on memory bandwidth, and yank out the
    // instructions array in between using it for the then block and else block.
    var sub_block = parent_block.makeSubBlock();
    sub_block.runtime_loop = null;
    sub_block.runtime_cond = cond_src;
    sub_block.runtime_index += 1;
    defer sub_block.instructions.deinit(gpa);

    try sema.analyzeBody(&sub_block, then_body);
    const true_instructions = sub_block.instructions.toOwnedSlice(gpa);
    defer gpa.free(true_instructions);

    try sema.analyzeBody(&sub_block, else_body);
    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.CondBr).Struct.fields.len +
        true_instructions.len + sub_block.instructions.items.len);
    _ = try parent_block.addInst(.{
        .tag = .cond_br,
        .data = .{ .pl_op = .{
            .operand = cond,
            .payload = sema.addExtraAssumeCapacity(Air.CondBr{
                .then_body_len = @intCast(u32, true_instructions.len),
                .else_body_len = @intCast(u32, sub_block.instructions.items.len),
            }),
        } },
    });
    sema.air_extra.appendSliceAssumeCapacity(true_instructions);
    sema.air_extra.appendSliceAssumeCapacity(sub_block.instructions.items);
    return always_noreturn;
}

fn zirUnreachable(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].@"unreachable";
    const src = inst_data.src();
    try sema.requireRuntimeBlock(block, src);
    // TODO Add compile error for @optimizeFor occurring too late in a scope.
    try block.addUnreachable(src, inst_data.safety);
    return always_noreturn;
}

fn zirRetErrValue(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Zir.Inst.Index {
    const inst_data = sema.code.instructions.items(.data)[inst].str_tok;
    const err_name = inst_data.get(sema.code);
    const src = inst_data.src();

    // Return the error code from the function.
    const kv = try sema.mod.getErrorValue(err_name);
    const result_inst = try sema.addConstant(
        try Type.Tag.error_set_single.create(sema.arena, kv.key),
        try Value.Tag.@"error".create(sema.arena, .{ .name = kv.key }),
    );
    return sema.analyzeRet(block, result_inst, src);
}

fn zirRetTok(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_tok;
    const operand = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();

    return sema.analyzeRet(block, operand, src);
}

fn zirRetNode(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const src = inst_data.src();

    return sema.analyzeRet(block, operand, src);
}

fn zirRetLoad(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Zir.Inst.Index {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const ret_ptr = sema.resolveInst(inst_data.operand);

    if (block.is_comptime or block.inlining != null) {
        const operand = try sema.analyzeLoad(block, src, ret_ptr, src);
        return sema.analyzeRet(block, operand, src);
    }
    try sema.requireRuntimeBlock(block, src);
    _ = try block.addUnOp(.ret_load, ret_ptr);
    return always_noreturn;
}

fn addToInferredErrorSet(sema: *Sema, uncasted_operand: Air.Inst.Ref) !void {
    assert(sema.fn_ret_ty.zigTypeTag() == .ErrorUnion);

    if (sema.fn_ret_ty.errorUnionSet().castTag(.error_set_inferred)) |payload| {
        const op_ty = sema.typeOf(uncasted_operand);
        switch (op_ty.zigTypeTag()) {
            .ErrorSet => {
                try payload.data.addErrorSet(sema.gpa, op_ty);
            },
            .ErrorUnion => {
                try payload.data.addErrorSet(sema.gpa, op_ty.errorUnionSet());
            },
            else => {},
        }
    }
}

fn analyzeRet(
    sema: *Sema,
    block: *Block,
    uncasted_operand: Air.Inst.Ref,
    src: LazySrcLoc,
) CompileError!Zir.Inst.Index {
    // Special case for returning an error to an inferred error set; we need to
    // add the error tag to the inferred error set of the in-scope function, so
    // that the coercion below works correctly.
    if (sema.fn_ret_ty.zigTypeTag() == .ErrorUnion) {
        try sema.addToInferredErrorSet(uncasted_operand);
    }
    const operand = try sema.coerce(block, sema.fn_ret_ty, uncasted_operand, src);

    if (block.inlining) |inlining| {
        if (block.is_comptime) {
            inlining.comptime_result = operand;
            return error.ComptimeReturn;
        }
        // We are inlining a function call; rewrite the `ret` as a `break`.
        try inlining.merges.results.append(sema.gpa, operand);
        _ = try block.addBr(inlining.merges.block_inst, operand);
        return always_noreturn;
    }

    try sema.resolveTypeLayout(block, src, sema.fn_ret_ty);
    _ = try block.addUnOp(.ret, operand);
    return always_noreturn;
}

fn floatOpAllowed(tag: Zir.Inst.Tag) bool {
    // extend this swich as additional operators are implemented
    return switch (tag) {
        .add, .sub, .mul, .div, .div_exact, .div_trunc, .div_floor, .mod, .rem, .mod_rem => true,
        else => false,
    };
}

fn zirPtrTypeSimple(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].ptr_type_simple;
    const elem_type = try sema.resolveType(block, .unneeded, inst_data.elem_type);
    const target = sema.mod.getTarget();
    const ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = elem_type,
        .@"addrspace" = .generic,
        .mutable = inst_data.is_mutable,
        .@"allowzero" = inst_data.is_allowzero or inst_data.size == .C,
        .@"volatile" = inst_data.is_volatile,
        .size = inst_data.size,
    });
    return sema.addType(ty);
}

fn zirPtrType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const src: LazySrcLoc = .unneeded;
    const elem_ty_src: LazySrcLoc = .unneeded;
    const inst_data = sema.code.instructions.items(.data)[inst].ptr_type;
    const extra = sema.code.extraData(Zir.Inst.PtrType, inst_data.payload_index);
    const unresolved_elem_ty = try sema.resolveType(block, elem_ty_src, extra.data.elem_type);
    const target = sema.mod.getTarget();

    var extra_i = extra.end;

    const sentinel = if (inst_data.flags.has_sentinel) blk: {
        const ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_i]);
        extra_i += 1;
        break :blk (try sema.resolveInstConst(block, .unneeded, ref)).val;
    } else null;

    const abi_align: u32 = if (inst_data.flags.has_align) blk: {
        const ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_i]);
        extra_i += 1;
        const coerced = try sema.coerce(block, Type.u32, sema.resolveInst(ref), src);
        const val = try sema.resolveConstValue(block, src, coerced);
        // Check if this happens to be the lazy alignment of our element type, in
        // which case we can make this 0 without resolving it.
        if (val.castTag(.lazy_align)) |payload| {
            if (payload.data.eql(unresolved_elem_ty, target)) {
                break :blk 0;
            }
        }
        const abi_align = (try val.getUnsignedIntAdvanced(target, sema.kit(block, src))).?;
        break :blk @intCast(u32, abi_align);
    } else 0;

    const address_space = if (inst_data.flags.has_addrspace) blk: {
        const ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_i]);
        extra_i += 1;
        break :blk try sema.analyzeAddrspace(block, .unneeded, ref, .pointer);
    } else .generic;

    const bit_offset = if (inst_data.flags.has_bit_range) blk: {
        const ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_i]);
        extra_i += 1;
        const bit_offset = try sema.resolveInt(block, .unneeded, ref, Type.u16);
        break :blk @intCast(u16, bit_offset);
    } else 0;

    const host_size: u16 = if (inst_data.flags.has_bit_range) blk: {
        const ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_i]);
        extra_i += 1;
        const host_size = try sema.resolveInt(block, .unneeded, ref, Type.u16);
        break :blk @intCast(u16, host_size);
    } else 0;

    if (host_size != 0 and bit_offset >= host_size * 8) {
        return sema.fail(block, src, "bit offset starts after end of host integer", .{});
    }

    const elem_ty = if (abi_align == 0)
        unresolved_elem_ty
    else t: {
        const elem_ty = try sema.resolveTypeFields(block, elem_ty_src, unresolved_elem_ty);
        try sema.resolveTypeLayout(block, elem_ty_src, elem_ty);
        break :t elem_ty;
    };
    const ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = elem_ty,
        .sentinel = sentinel,
        .@"align" = abi_align,
        .@"addrspace" = address_space,
        .bit_offset = bit_offset,
        .host_size = host_size,
        .mutable = inst_data.flags.is_mutable,
        .@"allowzero" = inst_data.flags.is_allowzero,
        .@"volatile" = inst_data.flags.is_volatile,
        .size = inst_data.size,
    });
    return sema.addType(ty);
}

fn zirStructInitEmpty(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const obj_ty = try sema.resolveType(block, src, inst_data.operand);

    switch (obj_ty.zigTypeTag()) {
        .Struct => return structInitEmpty(sema, block, obj_ty, src, src),
        .Array => return arrayInitEmpty(sema, obj_ty),
        .Void => return sema.addConstant(obj_ty, Value.void),
        else => return sema.failWithArrayInitNotSupported(block, src, obj_ty),
    }
}

fn structInitEmpty(
    sema: *Sema,
    block: *Block,
    obj_ty: Type,
    dest_src: LazySrcLoc,
    init_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const gpa = sema.gpa;
    // This logic must be synchronized with that in `zirStructInit`.
    const struct_ty = try sema.resolveTypeFields(block, dest_src, obj_ty);
    const struct_obj = struct_ty.castTag(.@"struct").?.data;

    // The init values to use for the struct instance.
    const field_inits = try gpa.alloc(Air.Inst.Ref, struct_obj.fields.count());
    defer gpa.free(field_inits);

    var root_msg: ?*Module.ErrorMsg = null;

    for (struct_obj.fields.values()) |field, i| {
        if (field.default_val.tag() == .unreachable_value) {
            const field_name = struct_obj.fields.keys()[i];
            const template = "missing struct field: {s}";
            const args = .{field_name};
            if (root_msg) |msg| {
                try sema.errNote(block, init_src, msg, template, args);
            } else {
                root_msg = try sema.errMsg(block, init_src, template, args);
            }
        } else {
            field_inits[i] = try sema.addConstant(field.ty, field.default_val);
        }
    }
    return sema.finishStructInit(block, dest_src, field_inits, root_msg, struct_obj, struct_ty, false);
}

fn arrayInitEmpty(sema: *Sema, obj_ty: Type) CompileError!Air.Inst.Ref {
    if (obj_ty.sentinel()) |sentinel| {
        const val = try Value.Tag.empty_array_sentinel.create(sema.arena, sentinel);
        return sema.addConstant(obj_ty, val);
    } else {
        return sema.addConstant(obj_ty, Value.initTag(.empty_array));
    }
}

fn zirUnionInit(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const field_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const init_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.UnionInit, inst_data.payload_index).data;
    const union_ty = try sema.resolveType(block, ty_src, extra.union_type);
    const field_name = try sema.resolveConstString(block, field_src, extra.field_name);
    const init = sema.resolveInst(extra.init);
    return sema.unionInit(block, init, init_src, union_ty, ty_src, field_name, field_src);
}

fn unionInit(
    sema: *Sema,
    block: *Block,
    uncasted_init: Air.Inst.Ref,
    init_src: LazySrcLoc,
    union_ty: Type,
    union_ty_src: LazySrcLoc,
    field_name: []const u8,
    field_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const field_index = try sema.unionFieldIndex(block, union_ty, field_name, field_src);
    const field = union_ty.unionFields().values()[field_index];
    const init = try sema.coerce(block, field.ty, uncasted_init, init_src);

    if (try sema.resolveMaybeUndefVal(block, init_src, init)) |init_val| {
        const tag_val = try Value.Tag.enum_field_index.create(sema.arena, field_index);
        return sema.addConstant(union_ty, try Value.Tag.@"union".create(sema.arena, .{
            .tag = tag_val,
            .val = init_val,
        }));
    }

    try sema.requireRuntimeBlock(block, init_src);
    _ = union_ty_src;
    try sema.queueFullTypeResolution(union_ty);
    return block.addUnionInit(union_ty, field_index, init);
}

fn zirStructInit(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_ref: bool,
) CompileError!Air.Inst.Ref {
    const gpa = sema.gpa;
    const zir_datas = sema.code.instructions.items(.data);
    const inst_data = zir_datas[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.StructInit, inst_data.payload_index);
    const src = inst_data.src();

    const first_item = sema.code.extraData(Zir.Inst.StructInit.Item, extra.end).data;
    const first_field_type_data = zir_datas[first_item.field_type].pl_node;
    const first_field_type_extra = sema.code.extraData(Zir.Inst.FieldType, first_field_type_data.payload_index).data;
    const unresolved_struct_type = try sema.resolveType(block, src, first_field_type_extra.container_type);
    const resolved_ty = try sema.resolveTypeFields(block, src, unresolved_struct_type);

    if (resolved_ty.castTag(.@"struct")) |struct_payload| {
        // This logic must be synchronized with that in `zirStructInitEmpty`.
        const struct_obj = struct_payload.data;

        // Maps field index to field_type index of where it was already initialized.
        // For making sure all fields are accounted for and no fields are duplicated.
        const found_fields = try gpa.alloc(Zir.Inst.Index, struct_obj.fields.count());
        defer gpa.free(found_fields);
        mem.set(Zir.Inst.Index, found_fields, 0);

        // The init values to use for the struct instance.
        const field_inits = try gpa.alloc(Air.Inst.Ref, struct_obj.fields.count());
        defer gpa.free(field_inits);

        var field_i: u32 = 0;
        var extra_index = extra.end;

        while (field_i < extra.data.fields_len) : (field_i += 1) {
            const item = sema.code.extraData(Zir.Inst.StructInit.Item, extra_index);
            extra_index = item.end;

            const field_type_data = zir_datas[item.data.field_type].pl_node;
            const field_src: LazySrcLoc = .{ .node_offset_back2tok = field_type_data.src_node };
            const field_type_extra = sema.code.extraData(Zir.Inst.FieldType, field_type_data.payload_index).data;
            const field_name = sema.code.nullTerminatedString(field_type_extra.name_start);
            const field_index = struct_obj.fields.getIndex(field_name) orelse
                return sema.failWithBadStructFieldAccess(block, struct_obj, field_src, field_name);
            if (found_fields[field_index] != 0) {
                const other_field_type = found_fields[field_index];
                const other_field_type_data = zir_datas[other_field_type].pl_node;
                const other_field_src: LazySrcLoc = .{ .node_offset_back2tok = other_field_type_data.src_node };
                const msg = msg: {
                    const msg = try sema.errMsg(block, field_src, "duplicate field", .{});
                    errdefer msg.destroy(gpa);
                    try sema.errNote(block, other_field_src, msg, "other field here", .{});
                    break :msg msg;
                };
                return sema.failWithOwnedErrorMsg(block, msg);
            }
            found_fields[field_index] = item.data.field_type;
            field_inits[field_index] = sema.resolveInst(item.data.init);
        }

        var root_msg: ?*Module.ErrorMsg = null;

        for (found_fields) |field_type_inst, i| {
            if (field_type_inst != 0) continue;

            // Check if the field has a default init.
            const field = struct_obj.fields.values()[i];
            if (field.default_val.tag() == .unreachable_value) {
                const field_name = struct_obj.fields.keys()[i];
                const template = "missing struct field: {s}";
                const args = .{field_name};
                if (root_msg) |msg| {
                    try sema.errNote(block, src, msg, template, args);
                } else {
                    root_msg = try sema.errMsg(block, src, template, args);
                }
            } else {
                field_inits[i] = try sema.addConstant(field.ty, field.default_val);
            }
        }
        return sema.finishStructInit(block, src, field_inits, root_msg, struct_obj, resolved_ty, is_ref);
    } else if (resolved_ty.zigTypeTag() == .Union) {
        if (extra.data.fields_len != 1) {
            return sema.fail(block, src, "union initialization expects exactly one field", .{});
        }

        const item = sema.code.extraData(Zir.Inst.StructInit.Item, extra.end);

        const field_type_data = zir_datas[item.data.field_type].pl_node;
        const field_src: LazySrcLoc = .{ .node_offset_back2tok = field_type_data.src_node };
        const field_type_extra = sema.code.extraData(Zir.Inst.FieldType, field_type_data.payload_index).data;
        const field_name = sema.code.nullTerminatedString(field_type_extra.name_start);
        const field_index = try sema.unionFieldIndex(block, resolved_ty, field_name, field_src);

        const init_inst = sema.resolveInst(item.data.init);
        if (try sema.resolveMaybeUndefVal(block, field_src, init_inst)) |val| {
            const tag_val = try Value.Tag.enum_field_index.create(sema.arena, field_index);
            return sema.addConstantMaybeRef(
                block,
                src,
                resolved_ty,
                try Value.Tag.@"union".create(sema.arena, .{ .tag = tag_val, .val = val }),
                is_ref,
            );
        }

        if (is_ref) {
            const alloc = try block.addTy(.alloc, resolved_ty);
            const field_ptr = try sema.unionFieldPtr(block, field_src, alloc, field_name, field_src, resolved_ty);
            try sema.storePtr(block, src, field_ptr, init_inst);
            return alloc;
        }

        try sema.requireRuntimeBlock(block, src);
        try sema.queueFullTypeResolution(resolved_ty);
        return block.addUnionInit(resolved_ty, field_index, init_inst);
    }
    unreachable;
}

fn finishStructInit(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    field_inits: []const Air.Inst.Ref,
    root_msg: ?*Module.ErrorMsg,
    struct_obj: *Module.Struct,
    struct_ty: Type,
    is_ref: bool,
) !Air.Inst.Ref {
    const gpa = sema.gpa;

    if (root_msg) |msg| {
        const fqn = try struct_obj.getFullyQualifiedName(gpa);
        defer gpa.free(fqn);
        try sema.mod.errNoteNonLazy(
            struct_obj.srcLoc(),
            msg,
            "struct '{s}' declared here",
            .{fqn},
        );
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const is_comptime = for (field_inits) |field_init| {
        if (!(try sema.isComptimeKnown(block, src, field_init))) {
            break false;
        }
    } else true;

    if (is_comptime) {
        const values = try sema.arena.alloc(Value, field_inits.len);
        for (field_inits) |field_init, i| {
            values[i] = (sema.resolveMaybeUndefVal(block, src, field_init) catch unreachable).?;
        }
        const struct_val = try Value.Tag.aggregate.create(sema.arena, values);
        return sema.addConstantMaybeRef(block, src, struct_ty, struct_val, is_ref);
    }

    if (is_ref) {
        const alloc = try block.addTy(.alloc, struct_ty);
        for (field_inits) |field_init, i_usize| {
            const i = @intCast(u32, i_usize);
            const field_src = src;
            const field_ptr = try sema.structFieldPtrByIndex(block, src, alloc, i, struct_obj, field_src);
            try sema.storePtr(block, src, field_ptr, field_init);
        }

        return alloc;
    }

    try sema.requireRuntimeBlock(block, src);
    try sema.queueFullTypeResolution(struct_ty);
    return block.addAggregateInit(struct_ty, field_inits);
}

fn zirStructInitAnon(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_ref: bool,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.StructInitAnon, inst_data.payload_index);
    const types = try sema.arena.alloc(Type, extra.data.fields_len);
    const values = try sema.arena.alloc(Value, types.len);
    const names = try sema.arena.alloc([]const u8, types.len);

    const opt_runtime_src = rs: {
        var runtime_src: ?LazySrcLoc = null;
        var extra_index = extra.end;
        for (types) |*field_ty, i| {
            const item = sema.code.extraData(Zir.Inst.StructInitAnon.Item, extra_index);
            extra_index = item.end;

            names[i] = sema.code.nullTerminatedString(item.data.field_name);
            const init = sema.resolveInst(item.data.init);
            field_ty.* = sema.typeOf(init);
            const init_src = src; // TODO better source location
            if (try sema.resolveMaybeUndefVal(block, init_src, init)) |init_val| {
                values[i] = init_val;
            } else {
                values[i] = Value.initTag(.unreachable_value);
                runtime_src = init_src;
            }
        }
        break :rs runtime_src;
    };

    const tuple_ty = try Type.Tag.anon_struct.create(sema.arena, .{
        .names = names,
        .types = types,
        .values = values,
    });

    const runtime_src = opt_runtime_src orelse {
        const tuple_val = try Value.Tag.aggregate.create(sema.arena, values);
        return sema.addConstantMaybeRef(block, src, tuple_ty, tuple_val, is_ref);
    };

    try sema.requireRuntimeBlock(block, runtime_src);

    if (is_ref) {
        const target = sema.mod.getTarget();
        const alloc_ty = try Type.ptr(sema.arena, target, .{
            .pointee_type = tuple_ty,
            .@"addrspace" = target_util.defaultAddressSpace(target, .local),
        });
        const alloc = try block.addTy(.alloc, alloc_ty);
        var extra_index = extra.end;
        for (types) |field_ty, i_usize| {
            const i = @intCast(u32, i_usize);
            const item = sema.code.extraData(Zir.Inst.StructInitAnon.Item, extra_index);
            extra_index = item.end;

            const field_ptr_ty = try Type.ptr(sema.arena, target, .{
                .mutable = true,
                .@"addrspace" = target_util.defaultAddressSpace(target, .local),
                .pointee_type = field_ty,
            });
            if (values[i].tag() == .unreachable_value) {
                const init = sema.resolveInst(item.data.init);
                const field_ptr = try block.addStructFieldPtr(alloc, i, field_ptr_ty);
                _ = try block.addBinOp(.store, field_ptr, init);
            }
        }

        return alloc;
    }

    const element_refs = try sema.arena.alloc(Air.Inst.Ref, types.len);
    var extra_index = extra.end;
    for (types) |_, i| {
        const item = sema.code.extraData(Zir.Inst.StructInitAnon.Item, extra_index);
        extra_index = item.end;
        element_refs[i] = sema.resolveInst(item.data.init);
    }

    return block.addAggregateInit(tuple_ty, element_refs);
}

fn zirArrayInit(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_ref: bool,
    is_sent: bool,
) CompileError!Air.Inst.Ref {
    const gpa = sema.gpa;
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();

    const extra = sema.code.extraData(Zir.Inst.MultiOp, inst_data.payload_index);
    const args = sema.code.refSlice(extra.end, extra.data.operands_len);
    assert(args.len != 0);

    const resolved_args = try gpa.alloc(Air.Inst.Ref, args.len);
    defer gpa.free(resolved_args);

    for (args) |arg, i| resolved_args[i] = sema.resolveInst(arg);

    const elem_ty = sema.typeOf(resolved_args[0]);
    const array_ty = blk: {
        if (!is_sent) {
            break :blk try Type.Tag.array.create(sema.arena, .{
                .len = resolved_args.len,
                .elem_type = elem_ty,
            });
        }

        const sentinel_ref = resolved_args[resolved_args.len - 1];
        const val = try sema.resolveConstValue(block, src, sentinel_ref);
        break :blk try Type.Tag.array_sentinel.create(sema.arena, .{
            .len = resolved_args.len - 1,
            .sentinel = val,
            .elem_type = elem_ty,
        });
    };

    const opt_runtime_src: ?LazySrcLoc = for (resolved_args) |arg| {
        const arg_src = src; // TODO better source location
        const comptime_known = try sema.isComptimeKnown(block, arg_src, arg);
        if (!comptime_known) break arg_src;
    } else null;

    const runtime_src = opt_runtime_src orelse {
        const elem_vals = try sema.arena.alloc(Value, resolved_args.len);

        for (resolved_args) |arg, i| {
            // We checked that all args are comptime above.
            elem_vals[i] = (sema.resolveMaybeUndefVal(block, src, arg) catch unreachable).?;
        }

        const array_val = try Value.Tag.aggregate.create(sema.arena, elem_vals);
        return sema.addConstantMaybeRef(block, src, array_ty, array_val, is_ref);
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    try sema.queueFullTypeResolution(elem_ty);

    if (is_ref) {
        const target = sema.mod.getTarget();
        const alloc_ty = try Type.ptr(sema.arena, target, .{
            .pointee_type = array_ty,
            .@"addrspace" = target_util.defaultAddressSpace(target, .local),
        });
        const alloc = try block.addTy(.alloc, alloc_ty);

        const elem_ptr_ty = try Type.ptr(sema.arena, target, .{
            .mutable = true,
            .@"addrspace" = target_util.defaultAddressSpace(target, .local),
            .pointee_type = elem_ty,
        });
        const elem_ptr_ty_ref = try sema.addType(elem_ptr_ty);

        for (resolved_args) |arg, i| {
            const index = try sema.addIntUnsigned(Type.usize, i);
            const elem_ptr = try block.addPtrElemPtrTypeRef(alloc, index, elem_ptr_ty_ref);
            _ = try block.addBinOp(.store, elem_ptr, arg);
        }
        return alloc;
    }

    return block.addAggregateInit(array_ty, resolved_args);
}

fn zirArrayInitAnon(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_ref: bool,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const extra = sema.code.extraData(Zir.Inst.MultiOp, inst_data.payload_index);
    const operands = sema.code.refSlice(extra.end, extra.data.operands_len);

    const types = try sema.arena.alloc(Type, operands.len);
    const values = try sema.arena.alloc(Value, operands.len);

    const opt_runtime_src = rs: {
        var runtime_src: ?LazySrcLoc = null;
        for (operands) |operand, i| {
            const elem = sema.resolveInst(operand);
            types[i] = sema.typeOf(elem);
            const operand_src = src; // TODO better source location
            if (try sema.resolveMaybeUndefVal(block, operand_src, elem)) |val| {
                values[i] = val;
            } else {
                values[i] = Value.initTag(.unreachable_value);
                runtime_src = operand_src;
            }
        }
        break :rs runtime_src;
    };

    const tuple_ty = try Type.Tag.tuple.create(sema.arena, .{
        .types = types,
        .values = values,
    });

    const runtime_src = opt_runtime_src orelse {
        const tuple_val = try Value.Tag.aggregate.create(sema.arena, values);
        return sema.addConstantMaybeRef(block, src, tuple_ty, tuple_val, is_ref);
    };

    try sema.requireRuntimeBlock(block, runtime_src);

    if (is_ref) {
        const target = sema.mod.getTarget();
        const alloc_ty = try Type.ptr(sema.arena, target, .{
            .pointee_type = tuple_ty,
            .@"addrspace" = target_util.defaultAddressSpace(target, .local),
        });
        const alloc = try block.addTy(.alloc, alloc_ty);
        for (operands) |operand, i_usize| {
            const i = @intCast(u32, i_usize);
            const field_ptr_ty = try Type.ptr(sema.arena, target, .{
                .mutable = true,
                .@"addrspace" = target_util.defaultAddressSpace(target, .local),
                .pointee_type = types[i],
            });
            if (values[i].tag() == .unreachable_value) {
                const field_ptr = try block.addStructFieldPtr(alloc, i, field_ptr_ty);
                _ = try block.addBinOp(.store, field_ptr, sema.resolveInst(operand));
            }
        }

        return alloc;
    }

    const element_refs = try sema.arena.alloc(Air.Inst.Ref, operands.len);
    for (operands) |operand, i| {
        element_refs[i] = sema.resolveInst(operand);
    }

    return block.addAggregateInit(tuple_ty, element_refs);
}

fn addConstantMaybeRef(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
    val: Value,
    is_ref: bool,
) !Air.Inst.Ref {
    if (!is_ref) return sema.addConstant(ty, val);

    var anon_decl = try block.startAnonDecl(src);
    defer anon_decl.deinit();
    const decl = try anon_decl.finish(
        try ty.copy(anon_decl.arena()),
        try val.copy(anon_decl.arena()),
        0, // default alignment
    );
    return sema.analyzeDeclRef(decl);
}

fn zirFieldTypeRef(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.FieldTypeRef, inst_data.payload_index).data;
    const ty_src = inst_data.src();
    const field_src = inst_data.src();
    const aggregate_ty = try sema.resolveType(block, ty_src, extra.container_type);
    const field_name = try sema.resolveConstString(block, field_src, extra.field_name);
    return sema.fieldType(block, aggregate_ty, field_name, field_src, ty_src);
}

fn zirFieldType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.FieldType, inst_data.payload_index).data;
    const ty_src = inst_data.src();
    const field_src = inst_data.src();
    const aggregate_ty = try sema.resolveType(block, ty_src, extra.container_type);
    if (aggregate_ty.tag() == .var_args_param) return sema.addType(aggregate_ty);
    const field_name = sema.code.nullTerminatedString(extra.name_start);
    return sema.fieldType(block, aggregate_ty, field_name, field_src, ty_src);
}

fn fieldType(
    sema: *Sema,
    block: *Block,
    aggregate_ty: Type,
    field_name: []const u8,
    field_src: LazySrcLoc,
    ty_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const resolved_ty = try sema.resolveTypeFields(block, ty_src, aggregate_ty);
    const target = sema.mod.getTarget();
    switch (resolved_ty.zigTypeTag()) {
        .Struct => {
            const struct_obj = resolved_ty.castTag(.@"struct").?.data;
            const field = struct_obj.fields.get(field_name) orelse
                return sema.failWithBadStructFieldAccess(block, struct_obj, field_src, field_name);
            return sema.addType(field.ty);
        },
        .Union => {
            const union_obj = resolved_ty.cast(Type.Payload.Union).?.data;
            const field = union_obj.fields.get(field_name) orelse
                return sema.failWithBadUnionFieldAccess(block, union_obj, field_src, field_name);
            return sema.addType(field.ty);
        },
        else => return sema.fail(block, ty_src, "expected struct or union; found '{}'", .{
            resolved_ty.fmt(target),
        }),
    }
}

fn zirErrorReturnTrace(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    const unresolved_stack_trace_ty = try sema.getBuiltinType(block, src, "StackTrace");
    const stack_trace_ty = try sema.resolveTypeFields(block, src, unresolved_stack_trace_ty);
    const opt_stack_trace_ty = try Type.optional(sema.arena, stack_trace_ty);
    // https://github.com/ziglang/zig/issues/11259
    return sema.addConstant(opt_stack_trace_ty, Value.@"null");
}

fn zirFrame(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const src: LazySrcLoc = .{ .node_offset = @bitCast(i32, extended.operand) };
    return sema.fail(block, src, "TODO: Sema.zirFrame", .{});
}

fn zirAlignOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ty = try sema.resolveType(block, operand_src, inst_data.operand);
    const target = sema.mod.getTarget();
    return sema.addConstant(
        Type.comptime_int,
        try ty.lazyAbiAlignment(target, sema.arena),
    );
}

fn zirBoolToInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand = sema.resolveInst(inst_data.operand);
    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        if (val.isUndef()) return sema.addConstUndef(Type.initTag(.u1));
        const bool_ints = [2]Air.Inst.Ref{ .zero, .one };
        return bool_ints[@boolToInt(val.toBool())];
    }
    return block.addUnOp(.bool_to_int, operand);
}

fn zirErrorName(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    _ = src;
    const operand = sema.resolveInst(inst_data.operand);
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };

    if (try sema.resolveDefinedValue(block, operand_src, operand)) |val| {
        const bytes = val.castTag(.@"error").?.data.name;
        return sema.addStrLit(block, bytes);
    }

    // Similar to zirTagName, we have special AIR instruction for the error name in case an optimimzation pass
    // might be able to resolve the result at compile time.
    return block.addUnOp(.error_name, operand);
}

fn zirUnaryMath(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
    eval: fn (Value, Type, Allocator, std.Target) Allocator.Error!Value,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand = sema.resolveInst(inst_data.operand);
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    switch (operand_ty.zigTypeTag()) {
        .ComptimeFloat, .Float => {},
        .Vector => {
            const scalar_ty = operand_ty.scalarType();
            switch (scalar_ty.zigTypeTag()) {
                .ComptimeFloat, .Float => {},
                else => return sema.fail(block, operand_src, "expected vector of floats or float type, found '{}'", .{scalar_ty.fmt(target)}),
            }
        },
        else => return sema.fail(block, operand_src, "expected vector of floats or float type, found '{}'", .{operand_ty.fmt(target)}),
    }

    switch (operand_ty.zigTypeTag()) {
        .Vector => {
            const scalar_ty = operand_ty.scalarType();
            const vec_len = operand_ty.vectorLen();
            const result_ty = try Type.vector(sema.arena, vec_len, scalar_ty);
            if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef())
                    return sema.addConstUndef(result_ty);

                var elem_buf: Value.ElemValueBuffer = undefined;
                const elems = try sema.arena.alloc(Value, vec_len);
                for (elems) |*elem, i| {
                    const elem_val = val.elemValueBuffer(i, &elem_buf);
                    elem.* = try eval(elem_val, scalar_ty, sema.arena, target);
                }
                return sema.addConstant(
                    result_ty,
                    try Value.Tag.aggregate.create(sema.arena, elems),
                );
            }

            try sema.requireRuntimeBlock(block, operand_src);
            return block.addUnOp(air_tag, operand);
        },
        .ComptimeFloat, .Float => {
            if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |operand_val| {
                if (operand_val.isUndef())
                    return sema.addConstUndef(operand_ty);
                const result_val = try eval(operand_val, operand_ty, sema.arena, target);
                return sema.addConstant(operand_ty, result_val);
            }

            try sema.requireRuntimeBlock(block, operand_src);
            return block.addUnOp(air_tag, operand);
        },
        else => unreachable,
    }
}

fn zirTagName(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const src = inst_data.src();
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    try sema.resolveTypeLayout(block, operand_src, operand_ty);
    const enum_ty = switch (operand_ty.zigTypeTag()) {
        .EnumLiteral => {
            const val = try sema.resolveConstValue(block, operand_src, operand);
            const bytes = val.castTag(.enum_literal).?.data;
            return sema.addStrLit(block, bytes);
        },
        .Enum => operand_ty,
        .Union => operand_ty.unionTagType() orelse {
            const decl = operand_ty.getOwnerDecl();
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "union '{s}' is untagged", .{
                    decl.name,
                });
                errdefer msg.destroy(sema.gpa);
                try sema.mod.errNoteNonLazy(decl.srcLoc(), msg, "declared here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        },
        else => return sema.fail(block, operand_src, "expected enum or union; found {}", .{
            operand_ty.fmt(target),
        }),
    };
    const enum_decl = enum_ty.getOwnerDecl();
    const casted_operand = try sema.coerce(block, enum_ty, operand, operand_src);
    if (try sema.resolveDefinedValue(block, operand_src, casted_operand)) |val| {
        const field_index = enum_ty.enumTagFieldIndex(val, target) orelse {
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "no field with value {} in enum '{s}'", .{
                    casted_operand, enum_decl.name,
                });
                errdefer msg.destroy(sema.gpa);
                try sema.mod.errNoteNonLazy(enum_decl.srcLoc(), msg, "declared here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        };
        const field_name = enum_ty.enumFieldName(field_index);
        return sema.addStrLit(block, field_name);
    }
    // In case the value is runtime-known, we have an AIR instruction for this instead
    // of trying to lower it in Sema because an optimization pass may result in the operand
    // being comptime-known, which would let us elide the `tag_name` AIR instruction.
    return block.addUnOp(.tag_name, casted_operand);
}

fn zirReify(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    const type_info_ty = try sema.resolveBuiltinTypeFields(block, src, "Type");
    const uncasted_operand = sema.resolveInst(inst_data.operand);
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const type_info = try sema.coerce(block, type_info_ty, uncasted_operand, operand_src);
    const val = try sema.resolveConstValue(block, operand_src, type_info);
    const union_val = val.cast(Value.Payload.Union).?.data;
    const tag_ty = type_info_ty.unionTagType().?;
    const target = sema.mod.getTarget();
    const tag_index = tag_ty.enumTagFieldIndex(union_val.tag, target).?;
    switch (@intToEnum(std.builtin.TypeId, tag_index)) {
        .Type => return Air.Inst.Ref.type_type,
        .Void => return Air.Inst.Ref.void_type,
        .Bool => return Air.Inst.Ref.bool_type,
        .NoReturn => return Air.Inst.Ref.noreturn_type,
        .ComptimeFloat => return Air.Inst.Ref.comptime_float_type,
        .ComptimeInt => return Air.Inst.Ref.comptime_int_type,
        .Undefined => return Air.Inst.Ref.undefined_type,
        .Null => return Air.Inst.Ref.null_type,
        .AnyFrame => return Air.Inst.Ref.anyframe_type,
        .EnumLiteral => return Air.Inst.Ref.enum_literal_type,
        .Int => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            const signedness_val = struct_val[0];
            const bits_val = struct_val[1];

            const signedness = signedness_val.toEnum(std.builtin.Signedness);
            const bits = @intCast(u16, bits_val.toUnsignedInt(target));
            const ty = switch (signedness) {
                .signed => try Type.Tag.int_signed.create(sema.arena, bits),
                .unsigned => try Type.Tag.int_unsigned.create(sema.arena, bits),
            };
            return sema.addType(ty);
        },
        .Vector => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            const len_val = struct_val[0];
            const child_val = struct_val[1];

            const len = len_val.toUnsignedInt(target);
            var buffer: Value.ToTypeBuffer = undefined;
            const child_ty = child_val.toType(&buffer);

            const ty = try Type.vector(sema.arena, len, try child_ty.copy(sema.arena));
            return sema.addType(ty);
        },
        .Float => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            // bits: comptime_int,
            const bits_val = struct_val[0];

            const bits = @intCast(u16, bits_val.toUnsignedInt(target));
            const ty = switch (bits) {
                16 => Type.@"f16",
                32 => Type.@"f32",
                64 => Type.@"f64",
                80 => Type.@"f80",
                128 => Type.@"f128",
                else => return sema.fail(block, src, "{}-bit float unsupported", .{bits}),
            };
            return sema.addType(ty);
        },
        .Pointer => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            const size_val = struct_val[0];
            const is_const_val = struct_val[1];
            const is_volatile_val = struct_val[2];
            const alignment_val = struct_val[3];
            const address_space_val = struct_val[4];
            const child_val = struct_val[5];
            const is_allowzero_val = struct_val[6];
            const sentinel_val = struct_val[7];

            var buffer: Value.ToTypeBuffer = undefined;
            const child_ty = child_val.toType(&buffer);

            const ptr_size = size_val.toEnum(std.builtin.Type.Pointer.Size);

            var actual_sentinel: ?Value = null;
            if (!sentinel_val.isNull()) {
                if (ptr_size == .One or ptr_size == .C) {
                    return sema.fail(block, src, "sentinels are only allowed on slices and unknown-length pointers", .{});
                }
                const sentinel_ptr_val = sentinel_val.castTag(.opt_payload).?.data;
                const ptr_ty = try Type.ptr(sema.arena, target, .{
                    .@"addrspace" = .generic,
                    .pointee_type = child_ty,
                });
                actual_sentinel = (try sema.pointerDeref(block, src, sentinel_ptr_val, ptr_ty)).?;
            }

            const ty = try Type.ptr(sema.arena, target, .{
                .size = ptr_size,
                .mutable = !is_const_val.toBool(),
                .@"volatile" = is_volatile_val.toBool(),
                .@"align" = @intCast(u16, alignment_val.toUnsignedInt(target)), // TODO: Validate this value.
                .@"addrspace" = address_space_val.toEnum(std.builtin.AddressSpace),
                .pointee_type = try child_ty.copy(sema.arena),
                .@"allowzero" = is_allowzero_val.toBool(),
                .sentinel = actual_sentinel,
            });
            return sema.addType(ty);
        },
        .Array => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            // len: comptime_int,
            const len_val = struct_val[0];
            // child: type,
            const child_val = struct_val[1];
            // sentinel: ?*const anyopaque,
            const sentinel_val = struct_val[2];

            const len = len_val.toUnsignedInt(target);
            var buffer: Value.ToTypeBuffer = undefined;
            const child_ty = try child_val.toType(&buffer).copy(sema.arena);
            const sentinel = if (sentinel_val.castTag(.opt_payload)) |p| blk: {
                const ptr_ty = try Type.ptr(sema.arena, target, .{
                    .@"addrspace" = .generic,
                    .pointee_type = child_ty,
                });
                break :blk (try sema.pointerDeref(block, src, p.data, ptr_ty)).?;
            } else null;

            const ty = try Type.array(sema.arena, len, sentinel, child_ty, target);
            return sema.addType(ty);
        },
        .Optional => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            // child: type,
            const child_val = struct_val[0];

            var buffer: Value.ToTypeBuffer = undefined;
            const child_ty = try child_val.toType(&buffer).copy(sema.arena);

            const ty = try Type.optional(sema.arena, child_ty);
            return sema.addType(ty);
        },
        .ErrorUnion => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            // error_set: type,
            const error_set_val = struct_val[0];
            // payload: type,
            const payload_val = struct_val[1];

            var buffer: Value.ToTypeBuffer = undefined;
            const error_set_ty = try error_set_val.toType(&buffer).copy(sema.arena);
            const payload_ty = try payload_val.toType(&buffer).copy(sema.arena);

            const ty = try Type.Tag.error_union.create(sema.arena, .{
                .error_set = error_set_ty,
                .payload = payload_ty,
            });
            return sema.addType(ty);
        },
        .ErrorSet => {
            const payload_val = union_val.val.optionalValue() orelse
                return sema.addType(Type.initTag(.anyerror));
            const slice_val = payload_val.castTag(.slice).?.data;
            const decl = slice_val.ptr.pointerDecl().?;
            try sema.ensureDeclAnalyzed(decl);
            const array_val = decl.val.castTag(.aggregate).?.data;

            var names: Module.ErrorSet.NameMap = .{};
            try names.ensureUnusedCapacity(sema.arena, array_val.len);
            for (array_val) |elem_val| {
                const struct_val = elem_val.castTag(.aggregate).?.data;
                // TODO use reflection instead of magic numbers here
                // error_set: type,
                const name_val = struct_val[0];
                const name_str = try name_val.toAllocatedBytes(Type.initTag(.const_slice_u8), sema.arena, target);

                const kv = try sema.mod.getErrorValue(name_str);
                names.putAssumeCapacityNoClobber(kv.key, {});
            }

            // names must be sorted
            Module.ErrorSet.sortNames(&names);
            const ty = try Type.Tag.error_set_merged.create(sema.arena, names);
            return sema.addType(ty);
        },
        .Struct => {
            // TODO use reflection instead of magic numbers here
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // layout: containerlayout,
            const layout_val = struct_val[0];
            // fields: []const enumfield,
            const fields_val = struct_val[1];
            // decls: []const declaration,
            const decls_val = struct_val[2];
            // is_tuple: bool,
            const is_tuple_val = struct_val[3];

            // Decls
            if (decls_val.sliceLen(target) > 0) {
                return sema.fail(block, src, "reified structs must have no decls", .{});
            }

            return if (is_tuple_val.toBool())
                try sema.reifyTuple(block, src, fields_val)
            else
                try sema.reifyStruct(block, inst, src, layout_val, fields_val);
        },
        .Enum => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // TODO use reflection instead of magic numbers here
            // layout: ContainerLayout,
            const layout_val = struct_val[0];
            // tag_type: type,
            const tag_type_val = struct_val[1];
            // fields: []const EnumField,
            const fields_val = struct_val[2];
            // decls: []const Declaration,
            const decls_val = struct_val[3];
            // is_exhaustive: bool,
            const is_exhaustive_val = struct_val[4];

            // enum layout is always auto
            const layout = layout_val.toEnum(std.builtin.Type.ContainerLayout);
            if (layout != .Auto) {
                return sema.fail(block, src, "reified enums must have a layout .Auto", .{});
            }

            // Decls
            if (decls_val.sliceLen(target) > 0) {
                return sema.fail(block, src, "reified enums must have no decls", .{});
            }

            const mod = sema.mod;
            const gpa = sema.gpa;
            var new_decl_arena = std.heap.ArenaAllocator.init(gpa);
            errdefer new_decl_arena.deinit();
            const new_decl_arena_allocator = new_decl_arena.allocator();

            // Define our empty enum decl
            const enum_obj = try new_decl_arena_allocator.create(Module.EnumFull);
            const enum_ty_payload = try new_decl_arena_allocator.create(Type.Payload.EnumFull);
            enum_ty_payload.* = .{
                .base = .{
                    .tag = if (!is_exhaustive_val.toBool())
                        .enum_nonexhaustive
                    else
                        .enum_full,
                },
                .data = enum_obj,
            };
            const enum_ty = Type.initPayload(&enum_ty_payload.base);
            const enum_val = try Value.Tag.ty.create(new_decl_arena_allocator, enum_ty);
            const type_name = try sema.createTypeName(block, .anon, "enum");
            const new_decl = try mod.createAnonymousDeclNamed(block, .{
                .ty = Type.type,
                .val = enum_val,
            }, type_name);
            new_decl.owns_tv = true;
            errdefer mod.abortAnonDecl(new_decl);

            // Enum tag type
            var buffer: Value.ToTypeBuffer = undefined;
            const int_tag_ty = try tag_type_val.toType(&buffer).copy(new_decl_arena_allocator);

            enum_obj.* = .{
                .owner_decl = new_decl,
                .tag_ty = int_tag_ty,
                .tag_ty_inferred = false,
                .fields = .{},
                .values = .{},
                .node_offset = src.node_offset,
                .namespace = .{
                    .parent = block.namespace,
                    .ty = enum_ty,
                    .file_scope = block.getFileScope(),
                },
            };

            // Fields
            const fields_len = try sema.usizeCast(block, src, fields_val.sliceLen(target));
            if (fields_len > 0) {
                try enum_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);
                try enum_obj.values.ensureTotalCapacityContext(new_decl_arena_allocator, fields_len, .{
                    .ty = enum_obj.tag_ty,
                    .target = target,
                });

                var i: usize = 0;
                while (i < fields_len) : (i += 1) {
                    const elem_val = try fields_val.elemValue(sema.arena, i);
                    const field_struct_val = elem_val.castTag(.aggregate).?.data;
                    // TODO use reflection instead of magic numbers here
                    // name: []const u8
                    const name_val = field_struct_val[0];
                    // value: comptime_int
                    const value_val = field_struct_val[1];

                    const field_name = try name_val.toAllocatedBytes(
                        Type.initTag(.const_slice_u8),
                        new_decl_arena_allocator,
                        target,
                    );

                    const gop = enum_obj.fields.getOrPutAssumeCapacity(field_name);
                    if (gop.found_existing) {
                        // TODO: better source location
                        return sema.fail(block, src, "duplicate enum tag {s}", .{field_name});
                    }

                    const copied_tag_val = try value_val.copy(new_decl_arena_allocator);
                    enum_obj.values.putAssumeCapacityNoClobberContext(copied_tag_val, {}, .{
                        .ty = enum_obj.tag_ty,
                        .target = target,
                    });
                }
            }

            try new_decl.finalizeNewArena(&new_decl_arena);
            return sema.analyzeDeclVal(block, src, new_decl);
        },
        .Opaque => {
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // decls: []const Declaration,
            const decls_val = struct_val[0];

            // Decls
            if (decls_val.sliceLen(target) > 0) {
                return sema.fail(block, src, "reified opaque must have no decls", .{});
            }

            const mod = sema.mod;
            var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
            errdefer new_decl_arena.deinit();
            const new_decl_arena_allocator = new_decl_arena.allocator();

            const opaque_obj = try new_decl_arena_allocator.create(Module.Opaque);
            const opaque_ty_payload = try new_decl_arena_allocator.create(Type.Payload.Opaque);
            opaque_ty_payload.* = .{
                .base = .{ .tag = .@"opaque" },
                .data = opaque_obj,
            };
            const opaque_ty = Type.initPayload(&opaque_ty_payload.base);
            const opaque_val = try Value.Tag.ty.create(new_decl_arena_allocator, opaque_ty);
            const type_name = try sema.createTypeName(block, .anon, "opaque");
            const new_decl = try mod.createAnonymousDeclNamed(block, .{
                .ty = Type.type,
                .val = opaque_val,
            }, type_name);
            new_decl.owns_tv = true;
            errdefer mod.abortAnonDecl(new_decl);

            opaque_obj.* = .{
                .owner_decl = new_decl,
                .node_offset = src.node_offset,
                .namespace = .{
                    .parent = block.namespace,
                    .ty = opaque_ty,
                    .file_scope = block.getFileScope(),
                },
            };

            try new_decl.finalizeNewArena(&new_decl_arena);
            return sema.analyzeDeclVal(block, src, new_decl);
        },
        .Union => {
            // TODO use reflection instead of magic numbers here
            const struct_val = union_val.val.castTag(.aggregate).?.data;
            // layout: containerlayout,
            const layout_val = struct_val[0];
            // tag_type: ?type,
            const tag_type_val = struct_val[1];
            // fields: []const enumfield,
            const fields_val = struct_val[2];
            // decls: []const declaration,
            const decls_val = struct_val[3];

            // Decls
            if (decls_val.sliceLen(target) > 0) {
                return sema.fail(block, src, "reified unions must have no decls", .{});
            }

            var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
            errdefer new_decl_arena.deinit();
            const new_decl_arena_allocator = new_decl_arena.allocator();

            const union_obj = try new_decl_arena_allocator.create(Module.Union);
            const type_tag: Type.Tag = if (!tag_type_val.isNull()) .union_tagged else .@"union";
            const union_payload = try new_decl_arena_allocator.create(Type.Payload.Union);
            union_payload.* = .{
                .base = .{ .tag = type_tag },
                .data = union_obj,
            };
            const union_ty = Type.initPayload(&union_payload.base);
            const new_union_val = try Value.Tag.ty.create(new_decl_arena_allocator, union_ty);
            const type_name = try sema.createTypeName(block, .anon, "union");
            const new_decl = try sema.mod.createAnonymousDeclNamed(block, .{
                .ty = Type.type,
                .val = new_union_val,
            }, type_name);
            new_decl.owns_tv = true;
            errdefer sema.mod.abortAnonDecl(new_decl);
            union_obj.* = .{
                .owner_decl = new_decl,
                .tag_ty = Type.initTag(.@"null"),
                .fields = .{},
                .node_offset = src.node_offset,
                .zir_index = inst,
                .layout = layout_val.toEnum(std.builtin.Type.ContainerLayout),
                .status = .have_field_types,
                .namespace = .{
                    .parent = block.namespace,
                    .ty = union_ty,
                    .file_scope = block.getFileScope(),
                },
            };

            // Tag type
            const fields_len = try sema.usizeCast(block, src, fields_val.sliceLen(target));
            union_obj.tag_ty = if (tag_type_val.optionalValue()) |payload_val| blk: {
                var buffer: Value.ToTypeBuffer = undefined;
                break :blk try payload_val.toType(&buffer).copy(new_decl_arena_allocator);
            } else try sema.generateUnionTagTypeSimple(block, fields_len);

            // Fields
            if (fields_len > 0) {
                try union_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);

                var i: usize = 0;
                while (i < fields_len) : (i += 1) {
                    const elem_val = try fields_val.elemValue(sema.arena, i);
                    const field_struct_val = elem_val.castTag(.aggregate).?.data;
                    // TODO use reflection instead of magic numbers here
                    // name: []const u8
                    const name_val = field_struct_val[0];
                    // field_type: type,
                    const field_type_val = field_struct_val[1];
                    // alignment: comptime_int,
                    const alignment_val = field_struct_val[2];

                    const field_name = try name_val.toAllocatedBytes(
                        Type.initTag(.const_slice_u8),
                        new_decl_arena_allocator,
                        target,
                    );

                    const gop = union_obj.fields.getOrPutAssumeCapacity(field_name);
                    if (gop.found_existing) {
                        // TODO: better source location
                        return sema.fail(block, src, "duplicate union field {s}", .{field_name});
                    }

                    var buffer: Value.ToTypeBuffer = undefined;
                    gop.value_ptr.* = .{
                        .ty = try field_type_val.toType(&buffer).copy(new_decl_arena_allocator),
                        .abi_align = @intCast(u32, alignment_val.toUnsignedInt(target)),
                    };
                }
            }

            try new_decl.finalizeNewArena(&new_decl_arena);
            return sema.analyzeDeclVal(block, src, new_decl);
        },
        .Fn => return sema.fail(block, src, "TODO: Sema.zirReify for Fn", .{}),
        .BoundFn => @panic("TODO delete BoundFn from the language"),
        .Frame => @panic("TODO implement https://github.com/ziglang/zig/issues/10710"),
    }
}

fn reifyTuple(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    fields_val: Value,
) CompileError!Air.Inst.Ref {
    const target = sema.mod.getTarget();

    const fields_len = try sema.usizeCast(block, src, fields_val.sliceLen(target));
    if (fields_len == 0) return sema.addType(Type.initTag(.empty_struct_literal));

    const types = try sema.arena.alloc(Type, fields_len);
    const values = try sema.arena.alloc(Value, fields_len);

    var used_fields: std.AutoArrayHashMapUnmanaged(u32, void) = .{};
    defer used_fields.deinit(sema.gpa);
    try used_fields.ensureTotalCapacity(sema.gpa, fields_len);

    var i: usize = 0;
    while (i < fields_len) : (i += 1) {
        const elem_val = try fields_val.elemValue(sema.arena, i);
        const field_struct_val = elem_val.castTag(.aggregate).?.data;
        // TODO use reflection instead of magic numbers here
        // name: []const u8
        const name_val = field_struct_val[0];
        // field_type: type,
        const field_type_val = field_struct_val[1];
        //default_value: ?*const anyopaque,
        const default_value_val = field_struct_val[2];

        const field_name = try name_val.toAllocatedBytes(
            Type.initTag(.const_slice_u8),
            sema.arena,
            target,
        );

        const field_index = std.fmt.parseUnsigned(u32, field_name, 10) catch |err| {
            return sema.fail(
                block,
                src,
                "tuple cannot have non-numeric field '{s}': {}",
                .{ field_name, err },
            );
        };

        if (field_index >= fields_len) {
            return sema.fail(
                block,
                src,
                "tuple field {} exceeds tuple field count",
                .{field_index},
            );
        }

        const gop = used_fields.getOrPutAssumeCapacity(field_index);
        if (gop.found_existing) {
            // TODO: better source location
            return sema.fail(block, src, "duplicate tuple field {}", .{field_index});
        }

        const default_val = if (default_value_val.optionalValue()) |opt_val| blk: {
            const payload_val = if (opt_val.pointerDecl()) |opt_decl|
                opt_decl.val
            else
                opt_val;
            break :blk try payload_val.copy(sema.arena);
        } else Value.initTag(.unreachable_value);

        var buffer: Value.ToTypeBuffer = undefined;
        types[field_index] = try field_type_val.toType(&buffer).copy(sema.arena);
        values[field_index] = default_val;
    }

    const ty = try Type.Tag.tuple.create(sema.arena, .{
        .types = types,
        .values = values,
    });
    return sema.addType(ty);
}

fn reifyStruct(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    src: LazySrcLoc,
    layout_val: Value,
    fields_val: Value,
) CompileError!Air.Inst.Ref {
    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const struct_obj = try new_decl_arena_allocator.create(Module.Struct);
    const struct_ty = try Type.Tag.@"struct".create(new_decl_arena_allocator, struct_obj);
    const new_struct_val = try Value.Tag.ty.create(new_decl_arena_allocator, struct_ty);
    const type_name = try sema.createTypeName(block, .anon, "struct");
    const new_decl = try sema.mod.createAnonymousDeclNamed(block, .{
        .ty = Type.type,
        .val = new_struct_val,
    }, type_name);
    new_decl.owns_tv = true;
    errdefer sema.mod.abortAnonDecl(new_decl);
    struct_obj.* = .{
        .owner_decl = new_decl,
        .fields = .{},
        .node_offset = src.node_offset,
        .zir_index = inst,
        .layout = layout_val.toEnum(std.builtin.Type.ContainerLayout),
        .status = .have_field_types,
        .known_non_opv = false,
        .namespace = .{
            .parent = block.namespace,
            .ty = struct_ty,
            .file_scope = block.getFileScope(),
        },
    };

    const target = sema.mod.getTarget();

    // Fields
    const fields_len = try sema.usizeCast(block, src, fields_val.sliceLen(target));
    try struct_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);
    var i: usize = 0;
    while (i < fields_len) : (i += 1) {
        const elem_val = try fields_val.elemValue(sema.arena, i);
        const field_struct_val = elem_val.castTag(.aggregate).?.data;
        // TODO use reflection instead of magic numbers here
        // name: []const u8
        const name_val = field_struct_val[0];
        // field_type: type,
        const field_type_val = field_struct_val[1];
        //default_value: ?*const anyopaque,
        const default_value_val = field_struct_val[2];
        // is_comptime: bool,
        const is_comptime_val = field_struct_val[3];
        // alignment: comptime_int,
        const alignment_val = field_struct_val[4];

        const field_name = try name_val.toAllocatedBytes(
            Type.initTag(.const_slice_u8),
            new_decl_arena_allocator,
            target,
        );

        const gop = struct_obj.fields.getOrPutAssumeCapacity(field_name);
        if (gop.found_existing) {
            // TODO: better source location
            return sema.fail(block, src, "duplicate struct field {s}", .{field_name});
        }

        const default_val = if (default_value_val.optionalValue()) |opt_val| blk: {
            const payload_val = if (opt_val.pointerDecl()) |opt_decl|
                opt_decl.val
            else
                opt_val;
            break :blk try payload_val.copy(new_decl_arena_allocator);
        } else Value.initTag(.unreachable_value);

        var buffer: Value.ToTypeBuffer = undefined;
        gop.value_ptr.* = .{
            .ty = try field_type_val.toType(&buffer).copy(new_decl_arena_allocator),
            .abi_align = @intCast(u32, alignment_val.toUnsignedInt(target)),
            .default_val = default_val,
            .is_comptime = is_comptime_val.toBool(),
            .offset = undefined,
        };
    }

    try new_decl.finalizeNewArena(&new_decl_arena);
    return sema.analyzeDeclVal(block, src, new_decl);
}

fn zirTypeName(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ty = try sema.resolveType(block, ty_src, inst_data.operand);

    var anon_decl = try block.startAnonDecl(LazySrcLoc.unneeded);
    defer anon_decl.deinit();

    const target = sema.mod.getTarget();
    const bytes = try ty.nameAllocArena(anon_decl.arena(), target);

    const new_decl = try anon_decl.finish(
        try Type.Tag.array_u8_sentinel_0.create(anon_decl.arena(), bytes.len),
        try Value.Tag.bytes.create(anon_decl.arena(), bytes[0 .. bytes.len + 1]),
        0, // default alignment
    );

    return sema.analyzeDeclRef(new_decl);
}

fn zirFrameType(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    return sema.fail(block, src, "TODO: Sema.zirFrameType", .{});
}

fn zirFrameSize(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    return sema.fail(block, src, "TODO: Sema.zirFrameSize", .{});
}

fn zirFloatToInt(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const dest_ty = try sema.resolveType(block, ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);
    const operand_ty = sema.typeOf(operand);

    _ = try sema.checkIntType(block, ty_src, dest_ty);
    try sema.checkFloatType(block, operand_src, operand_ty);

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        const target = sema.mod.getTarget();
        const result_val = val.floatToInt(sema.arena, operand_ty, dest_ty, target) catch |err| switch (err) {
            error.FloatCannotFit => {
                return sema.fail(block, operand_src, "integer value {d} cannot be stored in type '{}'", .{
                    std.math.floor(val.toFloat(f64)),
                    dest_ty.fmt(target),
                });
            },
            else => |e| return e,
        };
        return sema.addConstant(dest_ty, result_val);
    }

    try sema.requireRuntimeBlock(block, operand_src);
    return block.addTyOp(.float_to_int, dest_ty, operand);
}

fn zirIntToFloat(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const dest_ty = try sema.resolveType(block, ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);
    const operand_ty = sema.typeOf(operand);

    try sema.checkFloatType(block, ty_src, dest_ty);
    _ = try sema.checkIntType(block, operand_src, operand_ty);

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        const target = sema.mod.getTarget();
        const result_val = try val.intToFloat(sema.arena, operand_ty, dest_ty, target);
        return sema.addConstant(dest_ty, result_val);
    }

    try sema.requireRuntimeBlock(block, operand_src);
    return block.addTyOp(.int_to_float, dest_ty, operand);
}

fn zirIntToPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();

    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;

    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand_res = sema.resolveInst(extra.rhs);
    const operand_coerced = try sema.coerce(block, Type.usize, operand_res, operand_src);

    const type_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const type_res = try sema.resolveType(block, src, extra.lhs);
    try sema.checkPtrType(block, type_src, type_res);
    try sema.resolveTypeLayout(block, src, type_res.elemType2());
    const ptr_align = type_res.ptrAlignment(sema.mod.getTarget());
    const target = sema.mod.getTarget();

    if (try sema.resolveDefinedValue(block, operand_src, operand_coerced)) |val| {
        const addr = val.toUnsignedInt(target);
        if (!type_res.isAllowzeroPtr() and addr == 0)
            return sema.fail(block, operand_src, "pointer type '{}' does not allow address zero", .{type_res.fmt(target)});
        if (addr != 0 and addr % ptr_align != 0)
            return sema.fail(block, operand_src, "pointer type '{}' requires aligned address", .{type_res.fmt(target)});

        const val_payload = try sema.arena.create(Value.Payload.U64);
        val_payload.* = .{
            .base = .{ .tag = .int_u64 },
            .data = addr,
        };
        return sema.addConstant(type_res, Value.initPayload(&val_payload.base));
    }

    try sema.requireRuntimeBlock(block, src);
    if (block.wantSafety()) {
        if (!type_res.isAllowzeroPtr()) {
            const is_non_zero = try block.addBinOp(.cmp_neq, operand_coerced, .zero_usize);
            try sema.addSafetyCheck(block, is_non_zero, .cast_to_null);
        }

        if (ptr_align > 1) {
            const val_payload = try sema.arena.create(Value.Payload.U64);
            val_payload.* = .{
                .base = .{ .tag = .int_u64 },
                .data = ptr_align - 1,
            };
            const align_minus_1 = try sema.addConstant(
                Type.usize,
                Value.initPayload(&val_payload.base),
            );
            const remainder = try block.addBinOp(.bit_and, operand_coerced, align_minus_1);
            const is_aligned = try block.addBinOp(.cmp_eq, remainder, .zero_usize);
            try sema.addSafetyCheck(block, is_aligned, .incorrect_alignment);
        }
    }
    return block.addBitCast(type_res, operand_coerced);
}

fn zirErrSetCast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();
    try sema.checkErrorSetType(block, dest_ty_src, dest_ty);
    try sema.checkErrorSetType(block, operand_src, operand_ty);

    if (try sema.resolveDefinedValue(block, operand_src, operand)) |val| {
        try sema.resolveInferredErrorSetTy(block, src, dest_ty);

        if (!dest_ty.isAnyError()) {
            const error_name = val.castTag(.@"error").?.data.name;
            if (!dest_ty.errorSetHasField(error_name)) {
                return sema.fail(
                    block,
                    src,
                    "error.{s} not a member of error set '{}'",
                    .{ error_name, dest_ty.fmt(target) },
                );
            }
        }

        return sema.addConstant(dest_ty, val);
    }

    try sema.requireRuntimeBlock(block, src);
    if (block.wantSafety()) {
        // TODO
    }

    return block.addBitCast(dest_ty, operand);
}

fn zirPtrCast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const dest_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    try sema.checkPtrType(block, dest_ty_src, dest_ty);
    try sema.checkPtrOperand(block, operand_src, operand_ty);
    if (dest_ty.isSlice()) {
        return sema.fail(block, dest_ty_src, "illegal pointer cast to slice", .{});
    }
    const ptr = if (operand_ty.isSlice())
        try sema.analyzeSlicePtr(block, operand_src, operand, operand_ty)
    else
        operand;

    try sema.resolveTypeLayout(block, dest_ty_src, dest_ty.elemType2());
    const dest_align = dest_ty.ptrAlignment(target);
    try sema.resolveTypeLayout(block, operand_src, operand_ty.elemType2());
    const operand_align = operand_ty.ptrAlignment(target);

    // If the destination is less aligned than the source, preserve the source alignment
    var aligned_dest_ty = if (operand_align <= dest_align) dest_ty else blk: {
        // Unwrap the pointer (or pointer-like optional) type, set alignment, and re-wrap into result
        if (dest_ty.zigTypeTag() == .Optional) {
            var buf: Type.Payload.ElemType = undefined;
            var dest_ptr_info = dest_ty.optionalChild(&buf).ptrInfo().data;
            dest_ptr_info.@"align" = operand_align;
            break :blk try Type.optional(sema.arena, try Type.ptr(sema.arena, target, dest_ptr_info));
        } else {
            var dest_ptr_info = dest_ty.ptrInfo().data;
            dest_ptr_info.@"align" = operand_align;
            break :blk try Type.ptr(sema.arena, target, dest_ptr_info);
        }
    };

    return sema.coerceCompatiblePtrs(block, aligned_dest_ty, ptr, operand_src);
}

fn zirTruncate(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    const dest_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const dest_scalar_ty = try sema.resolveType(block, dest_ty_src, extra.lhs);
    const operand = sema.resolveInst(extra.rhs);
    const dest_is_comptime_int = try sema.checkIntType(block, dest_ty_src, dest_scalar_ty);
    const operand_scalar_ty = try sema.checkIntOrVectorAllowComptime(block, operand, operand_src);
    const operand_ty = sema.typeOf(operand);
    const is_vector = operand_ty.zigTypeTag() == .Vector;
    const dest_ty = if (is_vector)
        try Type.vector(sema.arena, operand_ty.vectorLen(), dest_scalar_ty)
    else
        dest_scalar_ty;

    if (dest_is_comptime_int) {
        return sema.coerce(block, dest_ty, operand, operand_src);
    }

    const target = sema.mod.getTarget();
    const dest_info = dest_scalar_ty.intInfo(target);

    if (try sema.typeHasOnePossibleValue(block, dest_ty_src, dest_ty)) |val| {
        return sema.addConstant(dest_ty, val);
    }

    if (operand_scalar_ty.zigTypeTag() != .ComptimeInt) {
        const operand_info = operand_ty.intInfo(target);
        if (try sema.typeHasOnePossibleValue(block, operand_src, operand_ty)) |val| {
            return sema.addConstant(operand_ty, val);
        }

        if (operand_info.signedness != dest_info.signedness) {
            return sema.fail(block, operand_src, "expected {s} integer type, found '{}'", .{
                @tagName(dest_info.signedness), operand_ty.fmt(target),
            });
        }
        if (operand_info.bits < dest_info.bits) {
            const msg = msg: {
                const msg = try sema.errMsg(
                    block,
                    src,
                    "destination type '{}' has more bits than source type '{}'",
                    .{ dest_ty.fmt(target), operand_ty.fmt(target) },
                );
                errdefer msg.destroy(sema.gpa);
                try sema.errNote(block, dest_ty_src, msg, "destination type has {d} bits", .{
                    dest_info.bits,
                });
                try sema.errNote(block, operand_src, msg, "operand type has {d} bits", .{
                    operand_info.bits,
                });
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
    }

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
        if (val.isUndef()) return sema.addConstUndef(dest_ty);
        if (!is_vector) {
            return sema.addConstant(
                dest_ty,
                try val.intTrunc(operand_ty, sema.arena, dest_info.signedness, dest_info.bits, target),
            );
        }
        var elem_buf: Value.ElemValueBuffer = undefined;
        const elems = try sema.arena.alloc(Value, operand_ty.vectorLen());
        for (elems) |*elem, i| {
            const elem_val = val.elemValueBuffer(i, &elem_buf);
            elem.* = try elem_val.intTrunc(operand_scalar_ty, sema.arena, dest_info.signedness, dest_info.bits, target);
        }
        return sema.addConstant(
            dest_ty,
            try Value.Tag.aggregate.create(sema.arena, elems),
        );
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.trunc, dest_ty, operand);
}

fn zirAlignCast(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const align_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const dest_align = try sema.resolveAlign(block, align_src, extra.lhs);
    const ptr = sema.resolveInst(extra.rhs);
    const ptr_ty = sema.typeOf(ptr);

    // TODO in addition to pointers, this instruction is supposed to work for
    // pointer-like optionals and slices.
    try sema.checkPtrOperand(block, ptr_src, ptr_ty);

    // TODO compile error if the result pointer is comptime known and would have an
    // alignment that disagrees with the Decl's alignment.

    // TODO insert safety check that the alignment is correct

    const ptr_info = ptr_ty.ptrInfo().data;
    const target = sema.mod.getTarget();
    const dest_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = ptr_info.pointee_type,
        .@"align" = dest_align,
        .@"addrspace" = ptr_info.@"addrspace",
        .mutable = ptr_info.mutable,
        .@"allowzero" = ptr_info.@"allowzero",
        .@"volatile" = ptr_info.@"volatile",
        .size = ptr_info.size,
    });
    return sema.coerceCompatiblePtrs(block, dest_ty, ptr, ptr_src);
}

fn zirBitCount(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
    comptimeOp: fn (val: Value, ty: Type, target: std.Target) u64,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    _ = try checkIntOrVector(sema, block, operand, operand_src);
    const target = sema.mod.getTarget();
    const bits = operand_ty.intInfo(target).bits;

    if (try sema.typeHasOnePossibleValue(block, operand_src, operand_ty)) |val| {
        return sema.addConstant(operand_ty, val);
    }

    const result_scalar_ty = try Type.smallestUnsignedInt(sema.arena, bits);
    switch (operand_ty.zigTypeTag()) {
        .Vector => {
            const vec_len = operand_ty.vectorLen();
            const result_ty = try Type.vector(sema.arena, vec_len, result_scalar_ty);
            if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef()) return sema.addConstUndef(result_ty);

                var elem_buf: Value.ElemValueBuffer = undefined;
                const elems = try sema.arena.alloc(Value, vec_len);
                const scalar_ty = operand_ty.scalarType();
                for (elems) |*elem, i| {
                    const elem_val = val.elemValueBuffer(i, &elem_buf);
                    const count = comptimeOp(elem_val, scalar_ty, target);
                    elem.* = try Value.Tag.int_u64.create(sema.arena, count);
                }
                return sema.addConstant(
                    result_ty,
                    try Value.Tag.aggregate.create(sema.arena, elems),
                );
            } else {
                try sema.requireRuntimeBlock(block, operand_src);
                return block.addTyOp(air_tag, result_ty, operand);
            }
        },
        .Int => {
            if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef()) return sema.addConstUndef(result_scalar_ty);
                return sema.addIntUnsigned(result_scalar_ty, comptimeOp(val, operand_ty, target));
            } else {
                try sema.requireRuntimeBlock(block, operand_src);
                return block.addTyOp(air_tag, result_scalar_ty, operand);
            }
        },
        else => unreachable,
    }
}

fn zirByteSwap(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    const scalar_ty = try sema.checkIntOrVectorAllowComptime(block, operand, operand_src);
    const target = sema.mod.getTarget();
    const bits = scalar_ty.intInfo(target).bits;
    if (bits % 8 != 0) {
        return sema.fail(
            block,
            ty_src,
            "@byteSwap requires the number of bits to be evenly divisible by 8, but {} has {} bits",
            .{ scalar_ty.fmt(target), bits },
        );
    }

    if (try sema.typeHasOnePossibleValue(block, operand_src, operand_ty)) |val| {
        return sema.addConstant(operand_ty, val);
    }

    switch (operand_ty.zigTypeTag()) {
        .Int, .ComptimeInt => {
            const runtime_src = if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef()) return sema.addConstUndef(operand_ty);
                const result_val = try val.byteSwap(operand_ty, target, sema.arena);
                return sema.addConstant(operand_ty, result_val);
            } else operand_src;

            try sema.requireRuntimeBlock(block, runtime_src);
            return block.addTyOp(.byte_swap, operand_ty, operand);
        },
        .Vector => {
            const runtime_src = if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef())
                    return sema.addConstUndef(operand_ty);

                const vec_len = operand_ty.vectorLen();
                var elem_buf: Value.ElemValueBuffer = undefined;
                const elems = try sema.arena.alloc(Value, vec_len);
                for (elems) |*elem, i| {
                    const elem_val = val.elemValueBuffer(i, &elem_buf);
                    elem.* = try elem_val.byteSwap(operand_ty, target, sema.arena);
                }
                return sema.addConstant(
                    operand_ty,
                    try Value.Tag.aggregate.create(sema.arena, elems),
                );
            } else operand_src;

            try sema.requireRuntimeBlock(block, runtime_src);
            return block.addTyOp(.byte_swap, operand_ty, operand);
        },
        else => unreachable,
    }
}

fn zirBitReverse(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand = sema.resolveInst(inst_data.operand);
    const operand_ty = sema.typeOf(operand);
    _ = try sema.checkIntOrVectorAllowComptime(block, operand, operand_src);

    if (try sema.typeHasOnePossibleValue(block, operand_src, operand_ty)) |val| {
        return sema.addConstant(operand_ty, val);
    }

    const target = sema.mod.getTarget();
    switch (operand_ty.zigTypeTag()) {
        .Int, .ComptimeInt => {
            const runtime_src = if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef()) return sema.addConstUndef(operand_ty);
                const result_val = try val.bitReverse(operand_ty, target, sema.arena);
                return sema.addConstant(operand_ty, result_val);
            } else operand_src;

            try sema.requireRuntimeBlock(block, runtime_src);
            return block.addTyOp(.bit_reverse, operand_ty, operand);
        },
        .Vector => {
            const runtime_src = if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |val| {
                if (val.isUndef())
                    return sema.addConstUndef(operand_ty);

                const vec_len = operand_ty.vectorLen();
                var elem_buf: Value.ElemValueBuffer = undefined;
                const elems = try sema.arena.alloc(Value, vec_len);
                for (elems) |*elem, i| {
                    const elem_val = val.elemValueBuffer(i, &elem_buf);
                    elem.* = try elem_val.bitReverse(operand_ty, target, sema.arena);
                }
                return sema.addConstant(
                    operand_ty,
                    try Value.Tag.aggregate.create(sema.arena, elems),
                );
            } else operand_src;

            try sema.requireRuntimeBlock(block, runtime_src);
            return block.addTyOp(.bit_reverse, operand_ty, operand);
        },
        else => unreachable,
    }
}

fn zirBitOffsetOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const offset = try bitOffsetOf(sema, block, inst);
    return sema.addIntUnsigned(Type.comptime_int, offset);
}

fn zirOffsetOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const offset = try bitOffsetOf(sema, block, inst);
    // TODO reminder to make this a compile error for packed structs
    return sema.addIntUnsigned(Type.comptime_int, offset / 8);
}

fn bitOffsetOf(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!u64 {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    sema.src = .{ .node_offset_bin_op = inst_data.src_node };
    const lhs_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;

    const ty = try sema.resolveType(block, lhs_src, extra.lhs);
    const field_name = try sema.resolveConstString(block, rhs_src, extra.rhs);
    const target = sema.mod.getTarget();

    try sema.resolveTypeLayout(block, lhs_src, ty);
    if (ty.tag() != .@"struct") {
        return sema.fail(
            block,
            lhs_src,
            "expected struct type, found '{}'",
            .{ty.fmt(target)},
        );
    }

    const fields = ty.structFields();
    const index = fields.getIndex(field_name) orelse {
        return sema.fail(
            block,
            rhs_src,
            "struct '{}' has no field '{s}'",
            .{ ty.fmt(target), field_name },
        );
    };

    switch (ty.containerLayout()) {
        .Packed => {
            var bit_sum: u64 = 0;
            for (fields.values()) |field, i| {
                if (i == index) {
                    return bit_sum;
                }
                bit_sum += field.ty.bitSize(target);
            } else unreachable;
        },
        else => {
            var it = ty.iterateStructOffsets(target);
            while (it.next()) |field_offset| {
                if (field_offset.field == index) {
                    return field_offset.offset * 8;
                }
            } else unreachable;
        },
    }
}

fn checkNamespaceType(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .Struct, .Enum, .Union, .Opaque => return,
        else => return sema.fail(block, src, "expected struct, enum, union, or opaque; found '{}'", .{ty.fmt(target)}),
    }
}

/// Returns `true` if the type was a comptime_int.
fn checkIntType(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!bool {
    const target = sema.mod.getTarget();
    switch (try ty.zigTypeTagOrPoison()) {
        .ComptimeInt => return true,
        .Int => return false,
        else => return sema.fail(block, src, "expected integer type, found '{}'", .{ty.fmt(target)}),
    }
}

fn checkPtrOperand(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .Pointer => return,
        .Fn => {
            const msg = msg: {
                const msg = try sema.errMsg(
                    block,
                    ty_src,
                    "expected pointer, found {}",
                    .{ty.fmt(target)},
                );
                errdefer msg.destroy(sema.gpa);

                try sema.errNote(block, ty_src, msg, "use '&' to obtain a function pointer", .{});

                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        },
        .Optional => if (ty.isPtrLikeOptional()) return,
        else => {},
    }
    return sema.fail(block, ty_src, "expected pointer type, found '{}'", .{ty.fmt(target)});
}

fn checkPtrType(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .Pointer => return,
        .Fn => {
            const msg = msg: {
                const msg = try sema.errMsg(
                    block,
                    ty_src,
                    "expected pointer type, found '{}'",
                    .{ty.fmt(target)},
                );
                errdefer msg.destroy(sema.gpa);

                try sema.errNote(block, ty_src, msg, "use '*const ' to make a function pointer type", .{});

                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        },
        .Optional => if (ty.isPtrLikeOptional()) return,
        else => {},
    }
    return sema.fail(block, ty_src, "expected pointer type, found '{}'", .{ty.fmt(target)});
}

fn checkVectorElemType(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    switch (ty.zigTypeTag()) {
        .Int, .Float, .Bool => return,
        else => if (ty.isPtrAtRuntime()) return,
    }
    const target = sema.mod.getTarget();
    return sema.fail(block, ty_src, "expected integer, float, bool, or pointer for the vector element type; found '{}'", .{ty.fmt(target)});
}

fn checkFloatType(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .ComptimeInt, .ComptimeFloat, .Float => {},
        else => return sema.fail(block, ty_src, "expected float type, found '{}'", .{ty.fmt(target)}),
    }
}

fn checkNumericType(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .ComptimeFloat, .Float, .ComptimeInt, .Int => {},
        .Vector => switch (ty.childType().zigTypeTag()) {
            .ComptimeFloat, .Float, .ComptimeInt, .Int => {},
            else => |t| return sema.fail(block, ty_src, "expected number, found '{}'", .{t}),
        },
        else => return sema.fail(block, ty_src, "expected number, found '{}'", .{ty.fmt(target)}),
    }
}

fn checkAtomicOperandType(
    sema: *Sema,
    block: *Block,
    ty_src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    var buffer: Type.Payload.Bits = undefined;
    const target = sema.mod.getTarget();
    const max_atomic_bits = target_util.largestAtomicBits(target);
    const int_ty = switch (ty.zigTypeTag()) {
        .Int => ty,
        .Enum => ty.intTagType(&buffer),
        .Float => {
            const bit_count = ty.floatBits(target);
            if (bit_count > max_atomic_bits) {
                return sema.fail(
                    block,
                    ty_src,
                    "expected {d}-bit float type or smaller; found {d}-bit float type",
                    .{ max_atomic_bits, bit_count },
                );
            }
            return;
        },
        .Bool => return, // Will be treated as `u8`.
        else => {
            if (ty.isPtrAtRuntime()) return;

            return sema.fail(
                block,
                ty_src,
                "expected bool, integer, float, enum, or pointer type; found {}",
                .{ty.fmt(target)},
            );
        },
    };
    const bit_count = int_ty.intInfo(target).bits;
    if (bit_count > max_atomic_bits) {
        return sema.fail(
            block,
            ty_src,
            "expected {d}-bit integer type or smaller; found {d}-bit integer type",
            .{ max_atomic_bits, bit_count },
        );
    }
}

fn checkPtrIsNotComptimeMutable(
    sema: *Sema,
    block: *Block,
    ptr_val: Value,
    ptr_src: LazySrcLoc,
    operand_src: LazySrcLoc,
) CompileError!void {
    _ = operand_src;
    if (ptr_val.isComptimeMutablePtr()) {
        return sema.fail(block, ptr_src, "cannot store runtime value in compile time variable", .{});
    }
}

fn checkComptimeVarStore(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    decl_ref_mut: Value.Payload.DeclRefMut.Data,
) CompileError!void {
    if (decl_ref_mut.runtime_index < block.runtime_index) {
        if (block.runtime_cond) |cond_src| {
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "store to comptime variable depends on runtime condition", .{});
                errdefer msg.destroy(sema.gpa);
                try sema.errNote(block, cond_src, msg, "runtime condition here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
        if (block.runtime_loop) |loop_src| {
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "cannot store to comptime variable in non-inline loop", .{});
                errdefer msg.destroy(sema.gpa);
                try sema.errNote(block, loop_src, msg, "non-inline loop here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
        unreachable;
    }
}

fn checkIntOrVector(
    sema: *Sema,
    block: *Block,
    operand: Air.Inst.Ref,
    operand_src: LazySrcLoc,
) CompileError!Type {
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();
    switch (try operand_ty.zigTypeTagOrPoison()) {
        .Int => return operand_ty,
        .Vector => {
            const elem_ty = operand_ty.childType();
            switch (try elem_ty.zigTypeTagOrPoison()) {
                .Int => return elem_ty,
                else => return sema.fail(block, operand_src, "expected vector of integers; found vector of '{}'", .{
                    elem_ty.fmt(target),
                }),
            }
        },
        else => return sema.fail(block, operand_src, "expected integer or vector, found '{}'", .{
            operand_ty.fmt(target),
        }),
    }
}

fn checkIntOrVectorAllowComptime(
    sema: *Sema,
    block: *Block,
    operand: Air.Inst.Ref,
    operand_src: LazySrcLoc,
) CompileError!Type {
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();
    switch (try operand_ty.zigTypeTagOrPoison()) {
        .Int, .ComptimeInt => return operand_ty,
        .Vector => {
            const elem_ty = operand_ty.childType();
            switch (try elem_ty.zigTypeTagOrPoison()) {
                .Int, .ComptimeInt => return elem_ty,
                else => return sema.fail(block, operand_src, "expected vector of integers; found vector of '{}'", .{
                    elem_ty.fmt(target),
                }),
            }
        },
        else => return sema.fail(block, operand_src, "expected integer or vector, found '{}'", .{
            operand_ty.fmt(target),
        }),
    }
}

fn checkErrorSetType(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!void {
    const target = sema.mod.getTarget();
    switch (ty.zigTypeTag()) {
        .ErrorSet => return,
        else => return sema.fail(block, src, "expected error set type, found '{}'", .{ty.fmt(target)}),
    }
}

const SimdBinOp = struct {
    len: ?usize,
    /// Coerced to `result_ty`.
    lhs: Air.Inst.Ref,
    /// Coerced to `result_ty`.
    rhs: Air.Inst.Ref,
    lhs_val: ?Value,
    rhs_val: ?Value,
    /// Only different than `scalar_ty` when it is a vector operation.
    result_ty: Type,
    scalar_ty: Type,
};

fn checkSimdBinOp(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    uncasted_lhs: Air.Inst.Ref,
    uncasted_rhs: Air.Inst.Ref,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!SimdBinOp {
    const lhs_ty = sema.typeOf(uncasted_lhs);
    const rhs_ty = sema.typeOf(uncasted_rhs);

    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);
    var vec_len: ?usize = if (lhs_ty.zigTypeTag() == .Vector) lhs_ty.vectorLen() else null;
    const result_ty = try sema.resolvePeerTypes(block, src, &.{ uncasted_lhs, uncasted_rhs }, .{
        .override = &[_]LazySrcLoc{ lhs_src, rhs_src },
    });
    const lhs = try sema.coerce(block, result_ty, uncasted_lhs, lhs_src);
    const rhs = try sema.coerce(block, result_ty, uncasted_rhs, rhs_src);

    return SimdBinOp{
        .len = vec_len,
        .lhs = lhs,
        .rhs = rhs,
        .lhs_val = try sema.resolveMaybeUndefVal(block, lhs_src, lhs),
        .rhs_val = try sema.resolveMaybeUndefVal(block, rhs_src, rhs),
        .result_ty = result_ty,
        .scalar_ty = result_ty.scalarType(),
    };
}

fn checkVectorizableBinaryOperands(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    lhs_ty: Type,
    rhs_ty: Type,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!void {
    const lhs_zig_ty_tag = try lhs_ty.zigTypeTagOrPoison();
    const rhs_zig_ty_tag = try rhs_ty.zigTypeTagOrPoison();
    if (lhs_zig_ty_tag != .Vector and rhs_zig_ty_tag != .Vector) return;

    const lhs_is_vector = switch (lhs_zig_ty_tag) {
        .Vector, .Array => true,
        else => false,
    };
    const rhs_is_vector = switch (rhs_zig_ty_tag) {
        .Vector, .Array => true,
        else => false,
    };

    if (lhs_is_vector and rhs_is_vector) {
        const lhs_len = lhs_ty.arrayLen();
        const rhs_len = rhs_ty.arrayLen();
        if (lhs_len != rhs_len) {
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "vector length mismatch", .{});
                errdefer msg.destroy(sema.gpa);
                try sema.errNote(block, lhs_src, msg, "length {d} here", .{lhs_len});
                try sema.errNote(block, rhs_src, msg, "length {d} here", .{rhs_len});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
    } else {
        const target = sema.mod.getTarget();
        const msg = msg: {
            const msg = try sema.errMsg(block, src, "mixed scalar and vector operands: {} and {}", .{
                lhs_ty.fmt(target), rhs_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);
            if (lhs_is_vector) {
                try sema.errNote(block, lhs_src, msg, "vector here", .{});
                try sema.errNote(block, rhs_src, msg, "scalar here", .{});
            } else {
                try sema.errNote(block, lhs_src, msg, "scalar here", .{});
                try sema.errNote(block, rhs_src, msg, "vector here", .{});
            }
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }
}

fn resolveExportOptions(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) CompileError!std.builtin.ExportOptions {
    const export_options_ty = try sema.getBuiltinType(block, src, "ExportOptions");
    const air_ref = sema.resolveInst(zir_ref);
    const options = try sema.coerce(block, export_options_ty, air_ref, src);

    const name = try sema.fieldVal(block, src, options, "name", src);
    const name_val = try sema.resolveConstValue(block, src, name);

    const linkage = try sema.fieldVal(block, src, options, "linkage", src);
    const linkage_val = try sema.resolveConstValue(block, src, linkage);

    const section = try sema.fieldVal(block, src, options, "section", src);
    const section_val = try sema.resolveConstValue(block, src, section);

    if (!section_val.isNull()) {
        return sema.fail(block, src, "TODO: implement exporting with linksection", .{});
    }
    const name_ty = Type.initTag(.const_slice_u8);
    const target = sema.mod.getTarget();
    return std.builtin.ExportOptions{
        .name = try name_val.toAllocatedBytes(name_ty, sema.arena, target),
        .linkage = linkage_val.toEnum(std.builtin.GlobalLinkage),
        .section = null, // TODO
    };
}

fn resolveBuiltinEnum(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
    comptime name: []const u8,
) CompileError!@field(std.builtin, name) {
    const ty = try sema.getBuiltinType(block, src, name);
    const air_ref = sema.resolveInst(zir_ref);
    const coerced = try sema.coerce(block, ty, air_ref, src);
    const val = try sema.resolveConstValue(block, src, coerced);
    return val.toEnum(@field(std.builtin, name));
}

fn resolveAtomicOrder(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) CompileError!std.builtin.AtomicOrder {
    return resolveBuiltinEnum(sema, block, src, zir_ref, "AtomicOrder");
}

fn resolveAtomicRmwOp(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
) CompileError!std.builtin.AtomicRmwOp {
    return resolveBuiltinEnum(sema, block, src, zir_ref, "AtomicRmwOp");
}

fn zirCmpxchg(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Cmpxchg, inst_data.payload_index).data;
    const src = inst_data.src();
    // zig fmt: off
    const elem_ty_src      : LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr_src          : LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const expected_src     : LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const new_value_src    : LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };
    const success_order_src: LazySrcLoc = .{ .node_offset_builtin_call_arg4 = inst_data.src_node };
    const failure_order_src: LazySrcLoc = .{ .node_offset_builtin_call_arg5 = inst_data.src_node };
    // zig fmt: on
    const ptr = sema.resolveInst(extra.ptr);
    const ptr_ty = sema.typeOf(ptr);
    const elem_ty = ptr_ty.elemType();
    try sema.checkAtomicOperandType(block, elem_ty_src, elem_ty);
    const target = sema.mod.getTarget();
    if (elem_ty.zigTypeTag() == .Float) {
        return sema.fail(
            block,
            elem_ty_src,
            "expected bool, integer, enum, or pointer type; found '{}'",
            .{elem_ty.fmt(target)},
        );
    }
    const expected_value = try sema.coerce(block, elem_ty, sema.resolveInst(extra.expected_value), expected_src);
    const new_value = try sema.coerce(block, elem_ty, sema.resolveInst(extra.new_value), new_value_src);
    const success_order = try sema.resolveAtomicOrder(block, success_order_src, extra.success_order);
    const failure_order = try sema.resolveAtomicOrder(block, failure_order_src, extra.failure_order);

    if (@enumToInt(success_order) < @enumToInt(std.builtin.AtomicOrder.Monotonic)) {
        return sema.fail(block, success_order_src, "success atomic ordering must be Monotonic or stricter", .{});
    }
    if (@enumToInt(failure_order) < @enumToInt(std.builtin.AtomicOrder.Monotonic)) {
        return sema.fail(block, failure_order_src, "failure atomic ordering must be Monotonic or stricter", .{});
    }
    if (@enumToInt(failure_order) > @enumToInt(success_order)) {
        return sema.fail(block, failure_order_src, "failure atomic ordering must be no stricter than success", .{});
    }
    if (failure_order == .Release or failure_order == .AcqRel) {
        return sema.fail(block, failure_order_src, "failure atomic ordering must not be Release or AcqRel", .{});
    }

    const result_ty = try Type.optional(sema.arena, elem_ty);

    // special case zero bit types
    if ((try sema.typeHasOnePossibleValue(block, elem_ty_src, elem_ty)) != null) {
        return sema.addConstant(result_ty, Value.@"null");
    }

    const runtime_src = if (try sema.resolveDefinedValue(block, ptr_src, ptr)) |ptr_val| rs: {
        if (try sema.resolveMaybeUndefVal(block, expected_src, expected_value)) |expected_val| {
            if (try sema.resolveMaybeUndefVal(block, new_value_src, new_value)) |new_val| {
                if (expected_val.isUndef() or new_val.isUndef()) {
                    // TODO: this should probably cause the memory stored at the pointer
                    // to become undef as well
                    return sema.addConstUndef(result_ty);
                }
                const stored_val = (try sema.pointerDeref(block, ptr_src, ptr_val, ptr_ty)) orelse break :rs ptr_src;
                const result_val = if (stored_val.eql(expected_val, elem_ty, target)) blk: {
                    try sema.storePtr(block, src, ptr, new_value);
                    break :blk Value.@"null";
                } else try Value.Tag.opt_payload.create(sema.arena, stored_val);

                return sema.addConstant(result_ty, result_val);
            } else break :rs new_value_src;
        } else break :rs expected_src;
    } else ptr_src;

    const flags: u32 = @as(u32, @enumToInt(success_order)) |
        (@as(u32, @enumToInt(failure_order)) << 3);

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addInst(.{
        .tag = air_tag,
        .data = .{ .ty_pl = .{
            .ty = try sema.addType(result_ty),
            .payload = try sema.addExtra(Air.Cmpxchg{
                .ptr = ptr,
                .expected_value = expected_value,
                .new_value = new_value,
                .flags = flags,
            }),
        } },
    });
}

fn zirSplat(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const len_src: LazySrcLoc = .{ .node_offset_bin_lhs = inst_data.src_node };
    const scalar_src: LazySrcLoc = .{ .node_offset_bin_rhs = inst_data.src_node };
    const len = @intCast(u32, try sema.resolveInt(block, len_src, extra.lhs, Type.u32));
    const scalar = sema.resolveInst(extra.rhs);
    const scalar_ty = sema.typeOf(scalar);
    try sema.checkVectorElemType(block, scalar_src, scalar_ty);
    const vector_ty = try Type.Tag.vector.create(sema.arena, .{
        .len = len,
        .elem_type = scalar_ty,
    });
    if (try sema.resolveMaybeUndefVal(block, scalar_src, scalar)) |scalar_val| {
        if (scalar_val.isUndef()) return sema.addConstUndef(vector_ty);

        return sema.addConstant(
            vector_ty,
            try Value.Tag.repeated.create(sema.arena, scalar_val),
        );
    }

    try sema.requireRuntimeBlock(block, scalar_src);
    return block.addTyOp(.splat, vector_ty, scalar);
}

fn zirReduce(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const op_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const operand_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operation = try sema.resolveBuiltinEnum(block, op_src, extra.lhs, "ReduceOp");
    const operand = sema.resolveInst(extra.rhs);
    const operand_ty = sema.typeOf(operand);
    const target = sema.mod.getTarget();

    if (operand_ty.zigTypeTag() != .Vector) {
        return sema.fail(block, operand_src, "expected vector, found {}", .{operand_ty.fmt(target)});
    }

    const scalar_ty = operand_ty.childType();

    // Type-check depending on operation.
    switch (operation) {
        .And, .Or, .Xor => switch (scalar_ty.zigTypeTag()) {
            .Int, .Bool => {},
            else => return sema.fail(block, operand_src, "@reduce operation '{s}' requires integer or boolean operand; found {}", .{
                @tagName(operation), operand_ty.fmt(target),
            }),
        },
        .Min, .Max, .Add, .Mul => switch (scalar_ty.zigTypeTag()) {
            .Int, .Float => {},
            else => return sema.fail(block, operand_src, "@reduce operation '{s}' requires integer or float operand; found {}", .{
                @tagName(operation), operand_ty.fmt(target),
            }),
        },
    }

    const vec_len = operand_ty.vectorLen();
    if (vec_len == 0) {
        // TODO re-evaluate if we should introduce a "neutral value" for some operations,
        // e.g. zero for add and one for mul.
        return sema.fail(block, operand_src, "@reduce operation requires a vector with nonzero length", .{});
    }

    if (try sema.resolveMaybeUndefVal(block, operand_src, operand)) |operand_val| {
        if (operand_val.isUndef()) return sema.addConstUndef(scalar_ty);

        var accum: Value = try operand_val.elemValue(sema.arena, 0);
        var elem_buf: Value.ElemValueBuffer = undefined;
        var i: u32 = 1;
        while (i < vec_len) : (i += 1) {
            const elem_val = operand_val.elemValueBuffer(i, &elem_buf);
            switch (operation) {
                .And => accum = try accum.bitwiseAnd(elem_val, scalar_ty, sema.arena, target),
                .Or => accum = try accum.bitwiseOr(elem_val, scalar_ty, sema.arena, target),
                .Xor => accum = try accum.bitwiseXor(elem_val, scalar_ty, sema.arena, target),
                .Min => accum = accum.numberMin(elem_val, target),
                .Max => accum = accum.numberMax(elem_val, target),
                .Add => accum = try accum.numberAddWrap(elem_val, scalar_ty, sema.arena, target),
                .Mul => accum = try accum.numberMulWrap(elem_val, scalar_ty, sema.arena, target),
            }
        }
        return sema.addConstant(scalar_ty, accum);
    }

    try sema.requireRuntimeBlock(block, operand_src);
    return block.addInst(.{
        .tag = .reduce,
        .data = .{ .reduce = .{
            .operand = operand,
            .operation = operation,
        } },
    });
}

fn zirShuffle(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Shuffle, inst_data.payload_index).data;
    const elem_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const mask_src: LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };

    const elem_ty = try sema.resolveType(block, elem_ty_src, extra.elem_type);
    try sema.checkVectorElemType(block, elem_ty_src, elem_ty);
    var a = sema.resolveInst(extra.a);
    var b = sema.resolveInst(extra.b);
    var mask = sema.resolveInst(extra.mask);
    var mask_ty = sema.typeOf(mask);
    const target = sema.mod.getTarget();

    const mask_len = switch (sema.typeOf(mask).zigTypeTag()) {
        .Array, .Vector => sema.typeOf(mask).arrayLen(),
        else => return sema.fail(block, mask_src, "expected vector or array, found {}", .{sema.typeOf(mask).fmt(target)}),
    };
    mask_ty = try Type.Tag.vector.create(sema.arena, .{
        .len = mask_len,
        .elem_type = Type.@"i32",
    });
    mask = try sema.coerce(block, mask_ty, mask, mask_src);
    const mask_val = try sema.resolveConstMaybeUndefVal(block, mask_src, mask);
    return sema.analyzeShuffle(block, inst_data.src_node, elem_ty, a, b, mask_val, @intCast(u32, mask_len));
}

fn analyzeShuffle(
    sema: *Sema,
    block: *Block,
    src_node: i32,
    elem_ty: Type,
    a_arg: Air.Inst.Ref,
    b_arg: Air.Inst.Ref,
    mask: Value,
    mask_len: u32,
) CompileError!Air.Inst.Ref {
    const a_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = src_node };
    const b_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = src_node };
    const mask_src: LazySrcLoc = .{ .node_offset_builtin_call_arg3 = src_node };
    var a = a_arg;
    var b = b_arg;

    const res_ty = try Type.Tag.vector.create(sema.arena, .{
        .len = mask_len,
        .elem_type = elem_ty,
    });

    const target = sema.mod.getTarget();
    var maybe_a_len = switch (sema.typeOf(a).zigTypeTag()) {
        .Array, .Vector => sema.typeOf(a).arrayLen(),
        .Undefined => null,
        else => return sema.fail(block, a_src, "expected vector or array with element type {}, found {}", .{
            elem_ty.fmt(target),
            sema.typeOf(a).fmt(target),
        }),
    };
    var maybe_b_len = switch (sema.typeOf(b).zigTypeTag()) {
        .Array, .Vector => sema.typeOf(b).arrayLen(),
        .Undefined => null,
        else => return sema.fail(block, b_src, "expected vector or array with element type {}, found {}", .{
            elem_ty.fmt(target),
            sema.typeOf(b).fmt(target),
        }),
    };
    if (maybe_a_len == null and maybe_b_len == null) {
        return sema.addConstUndef(res_ty);
    }
    const a_len = maybe_a_len orelse maybe_b_len.?;
    const b_len = maybe_b_len orelse a_len;

    const a_ty = try Type.Tag.vector.create(sema.arena, .{
        .len = a_len,
        .elem_type = elem_ty,
    });
    const b_ty = try Type.Tag.vector.create(sema.arena, .{
        .len = b_len,
        .elem_type = elem_ty,
    });

    if (maybe_a_len == null) a = try sema.addConstUndef(a_ty);
    if (maybe_b_len == null) b = try sema.addConstUndef(b_ty);

    const operand_info = [2]std.meta.Tuple(&.{ u64, LazySrcLoc, Type }){
        .{ a_len, a_src, a_ty },
        .{ b_len, b_src, b_ty },
    };

    var i: usize = 0;
    while (i < mask_len) : (i += 1) {
        var buf: Value.ElemValueBuffer = undefined;
        const elem = mask.elemValueBuffer(i, &buf);
        if (elem.isUndef()) continue;
        const int = elem.toSignedInt();
        var unsigned: u32 = undefined;
        var chosen: u32 = undefined;
        if (int >= 0) {
            unsigned = @intCast(u32, int);
            chosen = 0;
        } else {
            unsigned = @intCast(u32, ~int);
            chosen = 1;
        }
        if (unsigned >= operand_info[chosen][0]) {
            const msg = msg: {
                const msg = try sema.errMsg(block, mask_src, "mask index {d} has out-of-bounds selection", .{i});
                errdefer msg.destroy(sema.gpa);

                try sema.errNote(block, operand_info[chosen][1], msg, "selected index {d} out of bounds of {}", .{
                    unsigned,
                    operand_info[chosen][2].fmt(target),
                });

                if (chosen == 1) {
                    try sema.errNote(block, b_src, msg, "selections from the second vector are specified with negative numbers", .{});
                }

                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
    }

    if (try sema.resolveMaybeUndefVal(block, a_src, a)) |a_val| {
        if (try sema.resolveMaybeUndefVal(block, b_src, b)) |b_val| {
            const values = try sema.arena.alloc(Value, mask_len);

            i = 0;
            while (i < mask_len) : (i += 1) {
                var buf: Value.ElemValueBuffer = undefined;
                const mask_elem_val = mask.elemValueBuffer(i, &buf);
                if (mask_elem_val.isUndef()) {
                    values[i] = Value.undef;
                    continue;
                }
                const int = mask_elem_val.toSignedInt();
                const unsigned = if (int >= 0) @intCast(u32, int) else @intCast(u32, ~int);
                if (int >= 0) {
                    values[i] = try a_val.elemValue(sema.arena, unsigned);
                } else {
                    values[i] = try b_val.elemValue(sema.arena, unsigned);
                }
            }
            const res_val = try Value.Tag.aggregate.create(sema.arena, values);
            return sema.addConstant(res_ty, res_val);
        }
    }

    // All static analysis passed, and not comptime.
    // For runtime codegen, vectors a and b must be the same length. Here we
    // recursively @shuffle the smaller vector to append undefined elements
    // to it up to the length of the longer vector. This recursion terminates
    // in 1 call because these calls to analyzeShuffle guarantee a_len == b_len.
    if (a_len != b_len) {
        const min_len = std.math.min(a_len, b_len);
        const max_src = if (a_len > b_len) a_src else b_src;
        const max_len = try sema.usizeCast(block, max_src, std.math.max(a_len, b_len));

        const expand_mask_values = try sema.arena.alloc(Value, max_len);
        i = 0;
        while (i < min_len) : (i += 1) {
            expand_mask_values[i] = try Value.Tag.int_u64.create(sema.arena, i);
        }
        while (i < max_len) : (i += 1) {
            expand_mask_values[i] = Value.negative_one;
        }
        const expand_mask = try Value.Tag.aggregate.create(sema.arena, expand_mask_values);

        if (a_len < b_len) {
            const undef = try sema.addConstUndef(a_ty);
            a = try sema.analyzeShuffle(block, src_node, elem_ty, a, undef, expand_mask, @intCast(u32, max_len));
        } else {
            const undef = try sema.addConstUndef(b_ty);
            b = try sema.analyzeShuffle(block, src_node, elem_ty, b, undef, expand_mask, @intCast(u32, max_len));
        }
    }

    const mask_index = @intCast(u32, sema.air_values.items.len);
    try sema.air_values.append(sema.gpa, mask);
    return block.addInst(.{
        .tag = .shuffle,
        .data = .{ .ty_pl = .{
            .ty = try sema.addType(res_ty),
            .payload = try block.sema.addExtra(Air.Shuffle{
                .a = a,
                .b = b,
                .mask = mask_index,
                .mask_len = mask_len,
            }),
        } },
    });
}

fn zirSelect(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Select, inst_data.payload_index).data;
    const target = sema.mod.getTarget();

    const elem_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const pred_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const a_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const b_src: LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };

    const elem_ty = try sema.resolveType(block, elem_ty_src, extra.elem_type);
    try sema.checkVectorElemType(block, elem_ty_src, elem_ty);
    const pred_uncoerced = sema.resolveInst(extra.pred);
    const pred_ty = sema.typeOf(pred_uncoerced);

    const vec_len_u64 = switch (try pred_ty.zigTypeTagOrPoison()) {
        .Vector, .Array => pred_ty.arrayLen(),
        else => return sema.fail(block, pred_src, "expected vector or array, found '{}'", .{pred_ty.fmt(target)}),
    };
    const vec_len = try sema.usizeCast(block, pred_src, vec_len_u64);

    const bool_vec_ty = try Type.vector(sema.arena, vec_len, Type.bool);
    const pred = try sema.coerce(block, bool_vec_ty, pred_uncoerced, pred_src);

    const vec_ty = try Type.vector(sema.arena, vec_len, elem_ty);
    const a = try sema.coerce(block, vec_ty, sema.resolveInst(extra.a), a_src);
    const b = try sema.coerce(block, vec_ty, sema.resolveInst(extra.b), b_src);

    const maybe_pred = try sema.resolveMaybeUndefVal(block, pred_src, pred);
    const maybe_a = try sema.resolveMaybeUndefVal(block, a_src, a);
    const maybe_b = try sema.resolveMaybeUndefVal(block, b_src, b);

    const runtime_src = if (maybe_pred) |pred_val| rs: {
        if (pred_val.isUndef()) return sema.addConstUndef(vec_ty);

        if (maybe_a) |a_val| {
            if (a_val.isUndef()) return sema.addConstUndef(vec_ty);

            if (maybe_b) |b_val| {
                if (b_val.isUndef()) return sema.addConstUndef(vec_ty);

                var buf: Value.ElemValueBuffer = undefined;
                const elems = try sema.gpa.alloc(Value, vec_len);
                for (elems) |*elem, i| {
                    const pred_elem_val = pred_val.elemValueBuffer(i, &buf);
                    const should_choose_a = pred_elem_val.toBool();
                    if (should_choose_a) {
                        elem.* = a_val.elemValueBuffer(i, &buf);
                    } else {
                        elem.* = b_val.elemValueBuffer(i, &buf);
                    }
                }

                return sema.addConstant(
                    vec_ty,
                    try Value.Tag.aggregate.create(sema.arena, elems),
                );
            } else {
                break :rs b_src;
            }
        } else {
            if (maybe_b) |b_val| {
                if (b_val.isUndef()) return sema.addConstUndef(vec_ty);
            }
            break :rs a_src;
        }
    } else rs: {
        if (maybe_a) |a_val| {
            if (a_val.isUndef()) return sema.addConstUndef(vec_ty);
        }
        if (maybe_b) |b_val| {
            if (b_val.isUndef()) return sema.addConstUndef(vec_ty);
        }
        break :rs pred_src;
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addInst(.{
        .tag = .select,
        .data = .{ .pl_op = .{
            .operand = pred,
            .payload = try block.sema.addExtra(Air.Bin{
                .lhs = a,
                .rhs = b,
            }),
        } },
    });
}

fn zirAtomicLoad(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    // zig fmt: off
    const elem_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr_src    : LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const order_src  : LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    // zig fmt: on
    const ptr = sema.resolveInst(extra.lhs);
    const ptr_ty = sema.typeOf(ptr);
    const elem_ty = ptr_ty.elemType();
    try sema.checkAtomicOperandType(block, elem_ty_src, elem_ty);
    const order = try sema.resolveAtomicOrder(block, order_src, extra.rhs);

    switch (order) {
        .Release, .AcqRel => {
            return sema.fail(
                block,
                order_src,
                "@atomicLoad atomic ordering must not be Release or AcqRel",
                .{},
            );
        },
        else => {},
    }

    if (try sema.typeHasOnePossibleValue(block, elem_ty_src, elem_ty)) |val| {
        return sema.addConstant(elem_ty, val);
    }

    if (try sema.resolveDefinedValue(block, ptr_src, ptr)) |ptr_val| {
        if (try sema.pointerDeref(block, ptr_src, ptr_val, ptr_ty)) |elem_val| {
            return sema.addConstant(elem_ty, elem_val);
        }
    }

    try sema.requireRuntimeBlock(block, ptr_src);
    return block.addInst(.{
        .tag = .atomic_load,
        .data = .{ .atomic_load = .{
            .ptr = ptr,
            .order = order,
        } },
    });
}

fn zirAtomicRmw(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.AtomicRmw, inst_data.payload_index).data;
    const src = inst_data.src();
    // zig fmt: off
    const operand_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr_src       : LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const op_src        : LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const operand_src   : LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };
    const order_src     : LazySrcLoc = .{ .node_offset_builtin_call_arg4 = inst_data.src_node };
    // zig fmt: on
    const ptr = sema.resolveInst(extra.ptr);
    const ptr_ty = sema.typeOf(ptr);
    const operand_ty = ptr_ty.elemType();
    try sema.checkAtomicOperandType(block, operand_ty_src, operand_ty);
    const op = try sema.resolveAtomicRmwOp(block, op_src, extra.operation);

    switch (operand_ty.zigTypeTag()) {
        .Enum => if (op != .Xchg) {
            return sema.fail(block, op_src, "@atomicRmw with enum only allowed with .Xchg", .{});
        },
        .Bool => if (op != .Xchg) {
            return sema.fail(block, op_src, "@atomicRmw with bool only allowed with .Xchg", .{});
        },
        .Float => switch (op) {
            .Xchg, .Add, .Sub => {},
            else => return sema.fail(block, op_src, "@atomicRmw with float only allowed with .Xchg, .Add, and .Sub", .{}),
        },
        else => {},
    }
    const operand = try sema.coerce(block, operand_ty, sema.resolveInst(extra.operand), operand_src);
    const order = try sema.resolveAtomicOrder(block, order_src, extra.ordering);

    if (order == .Unordered) {
        return sema.fail(block, order_src, "@atomicRmw atomic ordering must not be Unordered", .{});
    }

    // special case zero bit types
    if (try sema.typeHasOnePossibleValue(block, operand_ty_src, operand_ty)) |val| {
        return sema.addConstant(operand_ty, val);
    }

    const runtime_src = if (try sema.resolveDefinedValue(block, ptr_src, ptr)) |ptr_val| rs: {
        const maybe_operand_val = try sema.resolveMaybeUndefVal(block, operand_src, operand);
        const operand_val = maybe_operand_val orelse {
            try sema.checkPtrIsNotComptimeMutable(block, ptr_val, ptr_src, operand_src);
            break :rs operand_src;
        };
        if (ptr_val.isComptimeMutablePtr()) {
            const target = sema.mod.getTarget();
            const stored_val = (try sema.pointerDeref(block, ptr_src, ptr_val, ptr_ty)) orelse break :rs ptr_src;
            const new_val = switch (op) {
                // zig fmt: off
                .Xchg => operand_val,
                .Add  => try stored_val.numberAddWrap(operand_val, operand_ty, sema.arena, target),
                .Sub  => try stored_val.numberSubWrap(operand_val, operand_ty, sema.arena, target),
                .And  => try stored_val.bitwiseAnd   (operand_val, operand_ty, sema.arena, target),
                .Nand => try stored_val.bitwiseNand  (operand_val, operand_ty, sema.arena, target),
                .Or   => try stored_val.bitwiseOr    (operand_val, operand_ty, sema.arena, target),
                .Xor  => try stored_val.bitwiseXor   (operand_val, operand_ty, sema.arena, target),
                .Max  =>     stored_val.numberMax    (operand_val,                         target),
                .Min  =>     stored_val.numberMin    (operand_val,                         target),
                // zig fmt: on
            };
            try sema.storePtrVal(block, src, ptr_val, new_val, operand_ty);
            return sema.addConstant(operand_ty, stored_val);
        } else break :rs ptr_src;
    } else ptr_src;

    const flags: u32 = @as(u32, @enumToInt(order)) | (@as(u32, @enumToInt(op)) << 3);

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addInst(.{
        .tag = .atomic_rmw,
        .data = .{ .pl_op = .{
            .operand = ptr,
            .payload = try sema.addExtra(Air.AtomicRmw{
                .operand = operand,
                .flags = flags,
            }),
        } },
    });
}

fn zirAtomicStore(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.AtomicStore, inst_data.payload_index).data;
    const src = inst_data.src();
    // zig fmt: off
    const operand_ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const ptr_src       : LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const operand_src   : LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const order_src     : LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };
    // zig fmt: on
    const ptr = sema.resolveInst(extra.ptr);
    const operand_ty = sema.typeOf(ptr).elemType();
    try sema.checkAtomicOperandType(block, operand_ty_src, operand_ty);
    const operand = try sema.coerce(block, operand_ty, sema.resolveInst(extra.operand), operand_src);
    const order = try sema.resolveAtomicOrder(block, order_src, extra.ordering);

    const air_tag: Air.Inst.Tag = switch (order) {
        .Acquire, .AcqRel => {
            return sema.fail(
                block,
                order_src,
                "@atomicStore atomic ordering must not be Acquire or AcqRel",
                .{},
            );
        },
        .Unordered => .atomic_store_unordered,
        .Monotonic => .atomic_store_monotonic,
        .Release => .atomic_store_release,
        .SeqCst => .atomic_store_seq_cst,
    };

    return sema.storePtr2(block, src, ptr, ptr_src, operand, operand_src, air_tag);
}

fn zirMulAdd(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.MulAdd, inst_data.payload_index).data;
    const src = inst_data.src();

    const mulend1_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const mulend2_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const addend_src: LazySrcLoc = .{ .node_offset_builtin_call_arg3 = inst_data.src_node };

    const addend = sema.resolveInst(extra.addend);
    const ty = sema.typeOf(addend);
    const mulend1 = try sema.coerce(block, ty, sema.resolveInst(extra.mulend1), mulend1_src);
    const mulend2 = try sema.coerce(block, ty, sema.resolveInst(extra.mulend2), mulend2_src);

    const target = sema.mod.getTarget();

    const maybe_mulend1 = try sema.resolveMaybeUndefVal(block, mulend1_src, mulend1);
    const maybe_mulend2 = try sema.resolveMaybeUndefVal(block, mulend2_src, mulend2);
    const maybe_addend = try sema.resolveMaybeUndefVal(block, addend_src, addend);

    switch (ty.zigTypeTag()) {
        .ComptimeFloat, .Float, .Vector => {},
        else => return sema.fail(block, src, "expected vector of floats or float type, found '{}'", .{ty.fmt(target)}),
    }

    const runtime_src = if (maybe_mulend1) |mulend1_val| rs: {
        if (maybe_mulend2) |mulend2_val| {
            if (mulend2_val.isUndef()) return sema.addConstUndef(ty);

            if (maybe_addend) |addend_val| {
                if (addend_val.isUndef()) return sema.addConstUndef(ty);
                const result_val = try Value.mulAdd(ty, mulend1_val, mulend2_val, addend_val, sema.arena, target);
                return sema.addConstant(ty, result_val);
            } else {
                break :rs addend_src;
            }
        } else {
            if (maybe_addend) |addend_val| {
                if (addend_val.isUndef()) return sema.addConstUndef(ty);
            }
            break :rs mulend2_src;
        }
    } else rs: {
        if (maybe_mulend2) |mulend2_val| {
            if (mulend2_val.isUndef()) return sema.addConstUndef(ty);
        }
        if (maybe_addend) |addend_val| {
            if (addend_val.isUndef()) return sema.addConstUndef(ty);
        }
        break :rs mulend1_src;
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addInst(.{
        .tag = .mul_add,
        .data = .{ .pl_op = .{
            .operand = addend,
            .payload = try sema.addExtra(Air.Bin{
                .lhs = mulend1,
                .rhs = mulend2,
            }),
        } },
    });
}

fn zirBuiltinCall(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const options_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const func_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const args_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const call_src = inst_data.src();

    const extra = sema.code.extraData(Zir.Inst.BuiltinCall, inst_data.payload_index);
    var func = sema.resolveInst(extra.data.callee);
    const options = sema.resolveInst(extra.data.options);
    const args = sema.resolveInst(extra.data.args);

    const modifier: std.builtin.CallOptions.Modifier = modifier: {
        const call_options_ty = try sema.getBuiltinType(block, options_src, "CallOptions");
        const coerced_options = try sema.coerce(block, call_options_ty, options, options_src);

        const modifier = try sema.fieldVal(block, options_src, coerced_options, "modifier", options_src);
        const modifier_val = try sema.resolveConstValue(block, options_src, modifier);

        const stack = try sema.fieldVal(block, options_src, coerced_options, "stack", options_src);
        const stack_val = try sema.resolveConstValue(block, options_src, stack);

        if (!stack_val.isNull()) {
            return sema.fail(block, options_src, "TODO: implement @call with stack", .{});
        }
        break :modifier modifier_val.toEnum(std.builtin.CallOptions.Modifier);
    };

    const target = sema.mod.getTarget();
    const args_ty = sema.typeOf(args);
    if (!args_ty.isTuple() and args_ty.tag() != .empty_struct_literal) {
        return sema.fail(block, args_src, "expected a tuple, found {}", .{args_ty.fmt(target)});
    }

    var resolved_args: []Air.Inst.Ref = undefined;

    // Desugar bound functions here
    if (sema.typeOf(func).tag() == .bound_fn) {
        const bound_func = try sema.resolveValue(block, func_src, func);
        const bound_data = &bound_func.cast(Value.Payload.BoundFn).?.data;
        func = bound_data.func_inst;
        resolved_args = try sema.arena.alloc(Air.Inst.Ref, args_ty.structFieldCount() + 1);
        resolved_args[0] = bound_data.arg0_inst;
        for (resolved_args[1..]) |*resolved, i| {
            resolved.* = try sema.tupleFieldValByIndex(block, args_src, args, @intCast(u32, i), args_ty);
        }
    } else {
        resolved_args = try sema.arena.alloc(Air.Inst.Ref, args_ty.structFieldCount());
        for (resolved_args) |*resolved, i| {
            resolved.* = try sema.tupleFieldValByIndex(block, args_src, args, @intCast(u32, i), args_ty);
        }
    }

    return sema.analyzeCall(block, func, func_src, call_src, modifier, false, resolved_args);
}

fn zirFieldParentPtr(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.FieldParentPtr, inst_data.payload_index).data;
    const src = inst_data.src();
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const ptr_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };

    const struct_ty = try sema.resolveType(block, ty_src, extra.parent_type);
    const field_name = try sema.resolveConstString(block, name_src, extra.field_name);
    const field_ptr = sema.resolveInst(extra.field_ptr);
    const field_ptr_ty = sema.typeOf(field_ptr);
    const target = sema.mod.getTarget();

    if (struct_ty.zigTypeTag() != .Struct) {
        return sema.fail(block, ty_src, "expected struct type, found '{}'", .{struct_ty.fmt(target)});
    }
    try sema.resolveTypeLayout(block, ty_src, struct_ty);

    const struct_obj = struct_ty.castTag(.@"struct").?.data;
    const field_index = struct_obj.fields.getIndex(field_name) orelse
        return sema.failWithBadStructFieldAccess(block, struct_obj, name_src, field_name);

    if (field_ptr_ty.zigTypeTag() != .Pointer) {
        return sema.fail(block, ty_src, "expected pointer type, found '{}'", .{field_ptr_ty.fmt(target)});
    }
    const field = struct_obj.fields.values()[field_index];
    const field_ptr_ty_info = field_ptr_ty.ptrInfo().data;

    var ptr_ty_data: Type.Payload.Pointer.Data = .{
        .pointee_type = field.ty,
        .mutable = field_ptr_ty_info.mutable,
        .@"addrspace" = field_ptr_ty_info.@"addrspace",
    };

    if (struct_obj.layout == .Packed) {
        return sema.fail(block, src, "TODO handle packed structs with @fieldParentPtr", .{});
    } else {
        ptr_ty_data.@"align" = field.abi_align;
    }

    const actual_field_ptr_ty = try Type.ptr(sema.arena, target, ptr_ty_data);
    const casted_field_ptr = try sema.coerce(block, actual_field_ptr_ty, field_ptr, ptr_src);

    ptr_ty_data.pointee_type = struct_ty;
    const result_ptr = try Type.ptr(sema.arena, target, ptr_ty_data);

    if (try sema.resolveDefinedValue(block, src, casted_field_ptr)) |field_ptr_val| {
        const payload = field_ptr_val.castTag(.field_ptr).?.data;
        return sema.addConstant(result_ptr, payload.container_ptr);
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addInst(.{
        .tag = .field_parent_ptr,
        .data = .{ .ty_pl = .{
            .ty = try sema.addType(result_ptr),
            .payload = try block.sema.addExtra(Air.FieldParentPtr{
                .field_ptr = casted_field_ptr,
                .field_index = @intCast(u32, field_index),
            }),
        } },
    });
}

fn zirMinMax(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    air_tag: Air.Inst.Tag,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Bin, inst_data.payload_index).data;
    const src = inst_data.src();
    const lhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const rhs_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const lhs = sema.resolveInst(extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    try sema.checkNumericType(block, lhs_src, sema.typeOf(lhs));
    try sema.checkNumericType(block, rhs_src, sema.typeOf(rhs));
    return sema.analyzeMinMax(block, src, lhs, rhs, air_tag, lhs_src, rhs_src);
}

fn analyzeMinMax(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
    air_tag: Air.Inst.Tag,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const simd_op = try sema.checkSimdBinOp(block, src, lhs, rhs, lhs_src, rhs_src);

    // TODO @maximum(max_int, undefined) should return max_int

    const runtime_src = if (simd_op.lhs_val) |lhs_val| rs: {
        if (lhs_val.isUndef()) return sema.addConstUndef(simd_op.result_ty);

        const rhs_val = simd_op.rhs_val orelse break :rs rhs_src;

        if (rhs_val.isUndef()) return sema.addConstUndef(simd_op.result_ty);

        const opFunc = switch (air_tag) {
            .min => Value.numberMin,
            .max => Value.numberMax,
            else => unreachable,
        };
        const target = sema.mod.getTarget();
        const vec_len = simd_op.len orelse {
            const result_val = opFunc(lhs_val, rhs_val, target);
            return sema.addConstant(simd_op.result_ty, result_val);
        };
        var lhs_buf: Value.ElemValueBuffer = undefined;
        var rhs_buf: Value.ElemValueBuffer = undefined;
        const elems = try sema.arena.alloc(Value, vec_len);
        for (elems) |*elem, i| {
            const lhs_elem_val = lhs_val.elemValueBuffer(i, &lhs_buf);
            const rhs_elem_val = rhs_val.elemValueBuffer(i, &rhs_buf);
            elem.* = opFunc(lhs_elem_val, rhs_elem_val, target);
        }
        return sema.addConstant(
            simd_op.result_ty,
            try Value.Tag.aggregate.create(sema.arena, elems),
        );
    } else rs: {
        if (simd_op.rhs_val) |rhs_val| {
            if (rhs_val.isUndef()) return sema.addConstUndef(simd_op.result_ty);
        }
        break :rs lhs_src;
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    return block.addBinOp(air_tag, simd_op.lhs, simd_op.rhs);
}

fn zirMemcpy(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Memcpy, inst_data.payload_index).data;
    const src = inst_data.src();
    const dest_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const src_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const len_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const dest_ptr = sema.resolveInst(extra.dest);
    const dest_ptr_ty = sema.typeOf(dest_ptr);
    const target = sema.mod.getTarget();

    try sema.checkPtrOperand(block, dest_src, dest_ptr_ty);
    if (dest_ptr_ty.isConstPtr()) {
        return sema.fail(block, dest_src, "cannot store through const pointer '{}'", .{dest_ptr_ty.fmt(target)});
    }

    const uncasted_src_ptr = sema.resolveInst(extra.source);
    const uncasted_src_ptr_ty = sema.typeOf(uncasted_src_ptr);
    try sema.checkPtrOperand(block, src_src, uncasted_src_ptr_ty);
    const src_ptr_info = uncasted_src_ptr_ty.ptrInfo().data;
    const wanted_src_ptr_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = dest_ptr_ty.elemType2(),
        .@"align" = src_ptr_info.@"align",
        .@"addrspace" = src_ptr_info.@"addrspace",
        .mutable = false,
        .@"allowzero" = src_ptr_info.@"allowzero",
        .@"volatile" = src_ptr_info.@"volatile",
        .size = .Many,
    });
    const src_ptr = try sema.coerce(block, wanted_src_ptr_ty, uncasted_src_ptr, src_src);
    const len = try sema.coerce(block, Type.usize, sema.resolveInst(extra.byte_count), len_src);

    const runtime_src = if (try sema.resolveDefinedValue(block, dest_src, dest_ptr)) |dest_ptr_val| rs: {
        if (!dest_ptr_val.isComptimeMutablePtr()) break :rs dest_src;
        if (try sema.resolveDefinedValue(block, src_src, src_ptr)) |src_ptr_val| {
            if (!src_ptr_val.isComptimeMutablePtr()) break :rs src_src;
            if (try sema.resolveDefinedValue(block, len_src, len)) |len_val| {
                _ = dest_ptr_val;
                _ = src_ptr_val;
                _ = len_val;
                return sema.fail(block, src, "TODO: Sema.zirMemcpy at comptime", .{});
            } else break :rs len_src;
        } else break :rs src_src;
    } else dest_src;

    try sema.requireRuntimeBlock(block, runtime_src);
    _ = try block.addInst(.{
        .tag = .memcpy,
        .data = .{ .pl_op = .{
            .operand = dest_ptr,
            .payload = try sema.addExtra(Air.Bin{
                .lhs = src_ptr,
                .rhs = len,
            }),
        } },
    });
}

fn zirMemset(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!void {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const extra = sema.code.extraData(Zir.Inst.Memset, inst_data.payload_index).data;
    const src = inst_data.src();
    const dest_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = inst_data.src_node };
    const value_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = inst_data.src_node };
    const len_src: LazySrcLoc = .{ .node_offset_builtin_call_arg2 = inst_data.src_node };
    const dest_ptr = sema.resolveInst(extra.dest);
    const dest_ptr_ty = sema.typeOf(dest_ptr);
    const target = sema.mod.getTarget();
    try sema.checkPtrOperand(block, dest_src, dest_ptr_ty);
    if (dest_ptr_ty.isConstPtr()) {
        return sema.fail(block, dest_src, "cannot store through const pointer '{}'", .{dest_ptr_ty.fmt(target)});
    }
    const elem_ty = dest_ptr_ty.elemType2();
    const value = try sema.coerce(block, elem_ty, sema.resolveInst(extra.byte), value_src);
    const len = try sema.coerce(block, Type.usize, sema.resolveInst(extra.byte_count), len_src);

    const runtime_src = if (try sema.resolveDefinedValue(block, dest_src, dest_ptr)) |ptr_val| rs: {
        if (!ptr_val.isComptimeMutablePtr()) break :rs dest_src;
        if (try sema.resolveDefinedValue(block, len_src, len)) |len_val| {
            if (try sema.resolveMaybeUndefVal(block, value_src, value)) |val| {
                _ = ptr_val;
                _ = len_val;
                _ = val;
                return sema.fail(block, src, "TODO: Sema.zirMemset at comptime", .{});
            } else break :rs value_src;
        } else break :rs len_src;
    } else dest_src;

    try sema.requireRuntimeBlock(block, runtime_src);
    _ = try block.addInst(.{
        .tag = .memset,
        .data = .{ .pl_op = .{
            .operand = dest_ptr,
            .payload = try sema.addExtra(Air.Bin{
                .lhs = value,
                .rhs = len,
            }),
        } },
    });
}

fn zirBuiltinAsyncCall(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].pl_node;
    const src = inst_data.src();
    return sema.fail(block, src, "TODO: Sema.zirBuiltinAsyncCall", .{});
}

fn zirResume(sema: *Sema, block: *Block, inst: Zir.Inst.Index) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();
    return sema.fail(block, src, "TODO: Sema.zirResume", .{});
}

fn zirAwait(
    sema: *Sema,
    block: *Block,
    inst: Zir.Inst.Index,
    is_nosuspend: bool,
) CompileError!Air.Inst.Ref {
    const inst_data = sema.code.instructions.items(.data)[inst].un_node;
    const src = inst_data.src();

    _ = is_nosuspend;
    return sema.fail(block, src, "TODO: Sema.zirAwait", .{});
}

fn zirVarExtended(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.ExtendedVar, extended.operand);
    const src = sema.src;
    const ty_src: LazySrcLoc = src; // TODO add a LazySrcLoc that points at type
    const mut_src: LazySrcLoc = src; // TODO add a LazySrcLoc that points at mut token
    const init_src: LazySrcLoc = src; // TODO add a LazySrcLoc that points at init expr
    const small = @bitCast(Zir.Inst.ExtendedVar.Small, extended.small);

    var extra_index: usize = extra.end;

    const lib_name: ?[]const u8 = if (small.has_lib_name) blk: {
        const lib_name = sema.code.nullTerminatedString(sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk lib_name;
    } else null;

    // ZIR supports encoding this information but it is not used; the information
    // is encoded via the Decl entry.
    assert(!small.has_align);
    //const align_val: Value = if (small.has_align) blk: {
    //    const align_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
    //    extra_index += 1;
    //    const align_tv = try sema.resolveInstConst(block, align_src, align_ref);
    //    break :blk align_tv.val;
    //} else Value.@"null";

    const uncasted_init: Air.Inst.Ref = if (small.has_init) blk: {
        const init_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk sema.resolveInst(init_ref);
    } else .none;

    const have_ty = extra.data.var_type != .none;
    const var_ty = if (have_ty)
        try sema.resolveType(block, ty_src, extra.data.var_type)
    else
        sema.typeOf(uncasted_init);

    const init_val = if (uncasted_init != .none) blk: {
        const init = if (have_ty)
            try sema.coerce(block, var_ty, uncasted_init, init_src)
        else
            uncasted_init;

        break :blk (try sema.resolveMaybeUndefVal(block, init_src, init)) orelse
            return sema.failWithNeededComptime(block, init_src);
    } else Value.initTag(.unreachable_value);

    try sema.validateVarType(block, mut_src, var_ty, small.is_extern);

    const new_var = try sema.gpa.create(Module.Var);
    errdefer sema.gpa.destroy(new_var);

    log.debug("created variable {*} owner_decl: {*} ({s})", .{
        new_var, sema.owner_decl, sema.owner_decl.name,
    });

    new_var.* = .{
        .owner_decl = sema.owner_decl,
        .init = init_val,
        .is_extern = small.is_extern,
        .is_mutable = true, // TODO get rid of this unused field
        .is_threadlocal = small.is_threadlocal,
        .is_weak_linkage = false,
        .lib_name = null,
    };

    if (lib_name) |lname| {
        new_var.lib_name = try sema.handleExternLibName(block, ty_src, lname);
    }

    const result = try sema.addConstant(
        var_ty,
        try Value.Tag.variable.create(sema.arena, new_var),
    );
    return result;
}

fn zirFuncExtended(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
    inst: Zir.Inst.Index,
) CompileError!Air.Inst.Ref {
    const tracy = trace(@src());
    defer tracy.end();

    const extra = sema.code.extraData(Zir.Inst.ExtendedFunc, extended.operand);
    const src: LazySrcLoc = .{ .node_offset = extra.data.src_node };
    const cc_src: LazySrcLoc = .{ .node_offset_fn_type_cc = extra.data.src_node };
    const align_src: LazySrcLoc = src; // TODO add a LazySrcLoc that points at align
    const small = @bitCast(Zir.Inst.ExtendedFunc.Small, extended.small);

    var extra_index: usize = extra.end;

    const lib_name: ?[]const u8 = if (small.has_lib_name) blk: {
        const lib_name = sema.code.nullTerminatedString(sema.code.extra[extra_index]);
        extra_index += 1;
        break :blk lib_name;
    } else null;

    const cc: std.builtin.CallingConvention = if (small.has_cc) blk: {
        const cc_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        const cc_tv = try sema.resolveInstConst(block, cc_src, cc_ref);
        break :blk cc_tv.val.toEnum(std.builtin.CallingConvention);
    } else .Unspecified;

    const align_val: Value = if (small.has_align) blk: {
        const align_ref = @intToEnum(Zir.Inst.Ref, sema.code.extra[extra_index]);
        extra_index += 1;
        const align_tv = try sema.resolveInstConst(block, align_src, align_ref);
        break :blk align_tv.val;
    } else Value.@"null";

    const ret_ty_body = sema.code.extra[extra_index..][0..extra.data.ret_body_len];
    extra_index += ret_ty_body.len;

    var src_locs: Zir.Inst.Func.SrcLocs = undefined;
    const has_body = extra.data.body_len != 0;
    if (has_body) {
        extra_index += extra.data.body_len;
        src_locs = sema.code.extraData(Zir.Inst.Func.SrcLocs, extra_index).data;
    }

    const is_var_args = small.is_var_args;
    const is_inferred_error = small.is_inferred_error;
    const is_extern = small.is_extern;

    return sema.funcCommon(
        block,
        extra.data.src_node,
        inst,
        ret_ty_body,
        cc,
        align_val,
        is_var_args,
        is_inferred_error,
        is_extern,
        has_body,
        src_locs,
        lib_name,
    );
}

fn zirCUndef(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.UnNode, extended.operand).data;
    const src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };

    const name = try sema.resolveConstString(block, src, extra.operand);
    try block.c_import_buf.?.writer().print("#undefine {s}\n", .{name});
    return Air.Inst.Ref.void_value;
}

fn zirCInclude(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.UnNode, extended.operand).data;
    const src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };

    const name = try sema.resolveConstString(block, src, extra.operand);
    try block.c_import_buf.?.writer().print("#include <{s}>\n", .{name});
    return Air.Inst.Ref.void_value;
}

fn zirCDefine(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.BinNode, extended.operand).data;
    const name_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const val_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = extra.node };

    const name = try sema.resolveConstString(block, name_src, extra.lhs);
    const rhs = sema.resolveInst(extra.rhs);
    if (sema.typeOf(rhs).zigTypeTag() != .Void) {
        const value = try sema.resolveConstString(block, val_src, extra.rhs);
        try block.c_import_buf.?.writer().print("#define {s} {s}\n", .{ name, value });
    } else {
        try block.c_import_buf.?.writer().print("#define {s}\n", .{name});
    }
    return Air.Inst.Ref.void_value;
}

fn zirWasmMemorySize(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.UnNode, extended.operand).data;
    const index_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const builtin_src: LazySrcLoc = .{ .node_offset = extra.node };
    const target = sema.mod.getTarget();
    if (!target.isWasm()) {
        return sema.fail(block, builtin_src, "builtin @wasmMemorySize is available when targeting WebAssembly; targeted CPU architecture is {s}", .{@tagName(target.cpu.arch)});
    }

    const index = @intCast(u32, try sema.resolveInt(block, index_src, extra.operand, Type.u32));
    try sema.requireRuntimeBlock(block, builtin_src);
    return block.addInst(.{
        .tag = .wasm_memory_size,
        .data = .{ .pl_op = .{
            .operand = .none,
            .payload = index,
        } },
    });
}

fn zirWasmMemoryGrow(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.BinNode, extended.operand).data;
    const builtin_src: LazySrcLoc = .{ .node_offset = extra.node };
    const index_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const delta_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = extra.node };
    const target = sema.mod.getTarget();
    if (!target.isWasm()) {
        return sema.fail(block, builtin_src, "builtin @wasmMemoryGrow is available when targeting WebAssembly; targeted CPU architecture is {s}", .{@tagName(target.cpu.arch)});
    }

    const index = @intCast(u32, try sema.resolveInt(block, index_src, extra.lhs, Type.u32));
    const delta = try sema.coerce(block, Type.u32, sema.resolveInst(extra.rhs), delta_src);

    try sema.requireRuntimeBlock(block, builtin_src);
    return block.addInst(.{
        .tag = .wasm_memory_grow,
        .data = .{ .pl_op = .{
            .operand = delta,
            .payload = index,
        } },
    });
}

fn zirPrefetch(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.BinNode, extended.operand).data;
    const ptr_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const opts_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = extra.node };
    const options_ty = try sema.getBuiltinType(block, opts_src, "PrefetchOptions");
    const ptr = sema.resolveInst(extra.lhs);
    try sema.checkPtrOperand(block, ptr_src, sema.typeOf(ptr));
    const options = try sema.coerce(block, options_ty, sema.resolveInst(extra.rhs), opts_src);
    const target = sema.mod.getTarget();

    const rw = try sema.fieldVal(block, opts_src, options, "rw", opts_src);
    const rw_val = try sema.resolveConstValue(block, opts_src, rw);
    const rw_tag = rw_val.toEnum(std.builtin.PrefetchOptions.Rw);

    const locality = try sema.fieldVal(block, opts_src, options, "locality", opts_src);
    const locality_val = try sema.resolveConstValue(block, opts_src, locality);
    const locality_int = @intCast(u2, locality_val.toUnsignedInt(target));

    const cache = try sema.fieldVal(block, opts_src, options, "cache", opts_src);
    const cache_val = try sema.resolveConstValue(block, opts_src, cache);
    const cache_tag = cache_val.toEnum(std.builtin.PrefetchOptions.Cache);

    if (!block.is_comptime) {
        _ = try block.addInst(.{
            .tag = .prefetch,
            .data = .{ .prefetch = .{
                .ptr = ptr,
                .rw = rw_tag,
                .locality = locality_int,
                .cache = cache_tag,
            } },
        });
    }

    return Air.Inst.Ref.void_value;
}

fn zirBuiltinExtern(
    sema: *Sema,
    block: *Block,
    extended: Zir.Inst.Extended.InstData,
) CompileError!Air.Inst.Ref {
    const extra = sema.code.extraData(Zir.Inst.BinNode, extended.operand).data;
    const src: LazySrcLoc = .{ .node_offset = extra.node };
    const ty_src: LazySrcLoc = .{ .node_offset_builtin_call_arg0 = extra.node };
    const options_src: LazySrcLoc = .{ .node_offset_builtin_call_arg1 = extra.node };

    var ty = try sema.resolveType(block, ty_src, extra.lhs);
    const options_inst = sema.resolveInst(extra.rhs);
    const target = sema.mod.getTarget();

    const options = options: {
        const extern_options_ty = try sema.getBuiltinType(block, options_src, "ExternOptions");
        const options = try sema.coerce(block, extern_options_ty, options_inst, options_src);

        const name = try sema.fieldVal(block, options_src, options, "name", options_src);
        const name_val = try sema.resolveConstValue(block, options_src, name);

        const library_name_inst = try sema.fieldVal(block, options_src, options, "library_name", options_src);
        const library_name_val = try sema.resolveConstValue(block, options_src, library_name_inst);

        const linkage = try sema.fieldVal(block, options_src, options, "linkage", options_src);
        const linkage_val = try sema.resolveConstValue(block, options_src, linkage);

        const is_thread_local = try sema.fieldVal(block, options_src, options, "is_thread_local", options_src);
        const is_thread_local_val = try sema.resolveConstValue(block, options_src, is_thread_local);

        var library_name: ?[]const u8 = null;
        if (!library_name_val.isNull()) {
            const payload = library_name_val.castTag(.opt_payload).?.data;
            library_name = try payload.toAllocatedBytes(Type.initTag(.const_slice_u8), sema.arena, target);
        }

        break :options std.builtin.ExternOptions{
            .name = try name_val.toAllocatedBytes(Type.initTag(.const_slice_u8), sema.arena, target),
            .library_name = library_name,
            .linkage = linkage_val.toEnum(std.builtin.GlobalLinkage),
            .is_thread_local = is_thread_local_val.toBool(),
        };
    };

    if (!ty.isPtrAtRuntime()) {
        return sema.fail(block, options_src, "expected (optional) pointer", .{});
    }

    if (options.name.len == 0) {
        return sema.fail(block, options_src, "extern symbol name cannot be empty", .{});
    }

    if (options.linkage != .Weak and options.linkage != .Strong) {
        return sema.fail(block, options_src, "extern symbol must use strong or weak linkage", .{});
    }

    if (options.linkage == .Weak and !ty.ptrAllowsZero()) {
        ty = try Type.optional(sema.arena, ty);
    }

    // TODO check duplicate extern

    const new_decl = try sema.mod.allocateNewDecl(try sema.gpa.dupeZ(u8, options.name), sema.owner_decl.src_namespace, sema.owner_decl.src_node, null);
    errdefer new_decl.destroy(sema.mod);

    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const new_var = try new_decl_arena_allocator.create(Module.Var);
    errdefer new_decl_arena_allocator.destroy(new_var);

    new_var.* = .{
        .owner_decl = sema.owner_decl,
        .init = Value.initTag(.unreachable_value),
        .is_extern = true,
        .is_mutable = false, // TODO get rid of this unused field
        .is_threadlocal = options.is_thread_local,
        .is_weak_linkage = options.linkage == .Weak,
        .lib_name = null,
    };

    if (options.library_name) |library_name| {
        if (library_name.len == 0) {
            return sema.fail(block, options_src, "library name name cannot be empty", .{});
        }
        new_var.lib_name = try sema.handleExternLibName(block, options_src, library_name);
    }

    new_decl.src_line = sema.owner_decl.src_line;
    new_decl.ty = try ty.copy(new_decl_arena_allocator);
    new_decl.val = try Value.Tag.variable.create(new_decl_arena_allocator, new_var);
    new_decl.@"align" = 0;
    new_decl.@"linksection" = null;
    new_decl.has_tv = true;
    new_decl.analysis = .complete;
    new_decl.generation = sema.mod.generation;

    const arena_state = try new_decl_arena_allocator.create(std.heap.ArenaAllocator.State);
    arena_state.* = new_decl_arena.state;
    new_decl.value_arena = arena_state;

    const ref = try sema.analyzeDeclRef(new_decl);
    try sema.requireRuntimeBlock(block, src);
    return block.addBitCast(ty, ref);
}

fn requireFunctionBlock(sema: *Sema, block: *Block, src: LazySrcLoc) !void {
    if (sema.func == null and !block.is_typeof) {
        return sema.fail(block, src, "instruction illegal outside function body", .{});
    }
}

fn requireRuntimeBlock(sema: *Sema, block: *Block, src: LazySrcLoc) !void {
    if (block.is_comptime) {
        return sema.failWithNeededComptime(block, src);
    }
    try sema.requireFunctionBlock(block, src);
}

/// Emit a compile error if type cannot be used for a runtime variable.
fn validateVarType(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    var_ty: Type,
    is_extern: bool,
) CompileError!void {
    if (try sema.validateRunTimeType(block, src, var_ty, is_extern)) return;

    const target = sema.mod.getTarget();
    const msg = msg: {
        const msg = try sema.errMsg(block, src, "variable of type '{}' must be const or comptime", .{var_ty.fmt(target)});
        errdefer msg.destroy(sema.gpa);

        try sema.explainWhyTypeIsComptime(block, src, msg, src.toSrcLoc(block.src_decl), var_ty);

        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn validateRunTimeType(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    var_ty: Type,
    is_extern: bool,
) CompileError!bool {
    var ty = var_ty;
    while (true) switch (ty.zigTypeTag()) {
        .Bool,
        .Int,
        .Float,
        .ErrorSet,
        .Enum,
        .Frame,
        .AnyFrame,
        .Void,
        => return true,

        .BoundFn,
        .ComptimeFloat,
        .ComptimeInt,
        .EnumLiteral,
        .NoReturn,
        .Type,
        .Undefined,
        .Null,
        .Fn,
        => return false,

        .Pointer => {
            const elem_ty = ty.childType();
            switch (elem_ty.zigTypeTag()) {
                .Opaque, .Fn => return true,
                else => ty = elem_ty,
            }
        },
        .Opaque => return is_extern,

        .Optional => {
            var buf: Type.Payload.ElemType = undefined;
            const child_ty = ty.optionalChild(&buf);
            return validateRunTimeType(sema, block, src, child_ty, is_extern);
        },
        .Array, .Vector => ty = ty.elemType(),

        .ErrorUnion => ty = ty.errorUnionPayload(),

        .Struct, .Union => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const needs_comptime = try sema.typeRequiresComptime(block, src, resolved_ty);
            return !needs_comptime;
        },
    };
}

fn explainWhyTypeIsComptime(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    msg: *Module.ErrorMsg,
    src_loc: Module.SrcLoc,
    ty: Type,
) CompileError!void {
    const mod = sema.mod;
    const target = mod.getTarget();
    switch (ty.zigTypeTag()) {
        .Bool,
        .Int,
        .Float,
        .ErrorSet,
        .Enum,
        .Frame,
        .AnyFrame,
        .Void,
        => return,

        .Fn => {
            try mod.errNoteNonLazy(src_loc, msg, "use '*const {}' for a function pointer type", .{
                ty.fmt(target),
            });
        },

        .Type => {
            try mod.errNoteNonLazy(src_loc, msg, "types are not available at runtime", .{});
        },

        .BoundFn,
        .ComptimeFloat,
        .ComptimeInt,
        .EnumLiteral,
        .NoReturn,
        .Undefined,
        .Null,
        .Opaque,
        .Optional,
        => return,

        .Pointer, .Array, .Vector => {
            try sema.explainWhyTypeIsComptime(block, src, msg, src_loc, ty.elemType());
        },

        .ErrorUnion => {
            try sema.explainWhyTypeIsComptime(block, src, msg, src_loc, ty.errorUnionPayload());
        },

        .Struct => {
            if (ty.castTag(.@"struct")) |payload| {
                const struct_obj = payload.data;
                for (struct_obj.fields.values()) |field, i| {
                    const field_src_loc = struct_obj.fieldSrcLoc(sema.gpa, .{
                        .index = i,
                        .range = .type,
                    });
                    if (try sema.typeRequiresComptime(block, src, field.ty)) {
                        try mod.errNoteNonLazy(field_src_loc, msg, "struct requires comptime because of this field", .{});
                        try sema.explainWhyTypeIsComptime(block, src, msg, field_src_loc, field.ty);
                    }
                }
            }
            // TODO tuples
        },

        .Union => {
            if (ty.cast(Type.Payload.Union)) |payload| {
                const union_obj = payload.data;
                for (union_obj.fields.values()) |field, i| {
                    const field_src_loc = union_obj.fieldSrcLoc(sema.gpa, .{
                        .index = i,
                        .range = .type,
                    });
                    if (try sema.typeRequiresComptime(block, src, field.ty)) {
                        try mod.errNoteNonLazy(field_src_loc, msg, "union requires comptime because of this field", .{});
                        try sema.explainWhyTypeIsComptime(block, src, msg, field_src_loc, field.ty);
                    }
                }
            }
        },
    }
}

pub const PanicId = enum {
    unreach,
    unwrap_null,
    unwrap_errunion,
    cast_to_null,
    incorrect_alignment,
    invalid_error_code,
    index_out_of_bounds,
    cast_truncated_data,
};

fn addSafetyCheck(
    sema: *Sema,
    parent_block: *Block,
    ok: Air.Inst.Ref,
    panic_id: PanicId,
) !void {
    const gpa = sema.gpa;

    var fail_block: Block = .{
        .parent = parent_block,
        .sema = sema,
        .src_decl = parent_block.src_decl,
        .namespace = parent_block.namespace,
        .wip_capture_scope = parent_block.wip_capture_scope,
        .instructions = .{},
        .inlining = parent_block.inlining,
        .is_comptime = parent_block.is_comptime,
    };

    defer fail_block.instructions.deinit(gpa);

    _ = try sema.safetyPanic(&fail_block, .unneeded, panic_id);

    try parent_block.instructions.ensureUnusedCapacity(gpa, 1);

    try sema.air_extra.ensureUnusedCapacity(gpa, @typeInfo(Air.Block).Struct.fields.len +
        1 + // The main block only needs space for the cond_br.
        @typeInfo(Air.CondBr).Struct.fields.len +
        1 + // The ok branch of the cond_br only needs space for the br.
        fail_block.instructions.items.len);

    try sema.air_instructions.ensureUnusedCapacity(gpa, 3);
    const block_inst = @intCast(Air.Inst.Index, sema.air_instructions.len);
    const cond_br_inst = block_inst + 1;
    const br_inst = cond_br_inst + 1;
    sema.air_instructions.appendAssumeCapacity(.{
        .tag = .block,
        .data = .{ .ty_pl = .{
            .ty = .void_type,
            .payload = sema.addExtraAssumeCapacity(Air.Block{
                .body_len = 1,
            }),
        } },
    });
    sema.air_extra.appendAssumeCapacity(cond_br_inst);

    sema.air_instructions.appendAssumeCapacity(.{
        .tag = .cond_br,
        .data = .{ .pl_op = .{
            .operand = ok,
            .payload = sema.addExtraAssumeCapacity(Air.CondBr{
                .then_body_len = 1,
                .else_body_len = @intCast(u32, fail_block.instructions.items.len),
            }),
        } },
    });
    sema.air_extra.appendAssumeCapacity(br_inst);
    sema.air_extra.appendSliceAssumeCapacity(fail_block.instructions.items);

    sema.air_instructions.appendAssumeCapacity(.{
        .tag = .br,
        .data = .{ .br = .{
            .block_inst = block_inst,
            .operand = .void_value,
        } },
    });

    parent_block.instructions.appendAssumeCapacity(block_inst);
}

fn panicWithMsg(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    msg_inst: Air.Inst.Ref,
) !Zir.Inst.Index {
    const mod = sema.mod;
    const arena = sema.arena;

    const this_feature_is_implemented_in_the_backend =
        mod.comp.bin_file.options.object_format == .c or
        mod.comp.bin_file.options.use_llvm;
    if (!this_feature_is_implemented_in_the_backend) {
        // TODO implement this feature in all the backends and then delete this branch
        _ = try block.addNoOp(.breakpoint);
        _ = try block.addNoOp(.unreach);
        return always_noreturn;
    }
    const panic_fn = try sema.getBuiltin(block, src, "panic");
    const unresolved_stack_trace_ty = try sema.getBuiltinType(block, src, "StackTrace");
    const stack_trace_ty = try sema.resolveTypeFields(block, src, unresolved_stack_trace_ty);
    const target = mod.getTarget();
    const ptr_stack_trace_ty = try Type.ptr(arena, target, .{
        .pointee_type = stack_trace_ty,
        .@"addrspace" = target_util.defaultAddressSpace(target, .global_constant), // TODO might need a place that is more dynamic
    });
    const null_stack_trace = try sema.addConstant(
        try Type.optional(arena, ptr_stack_trace_ty),
        Value.@"null",
    );
    const args = try arena.create([2]Air.Inst.Ref);
    args.* = .{ msg_inst, null_stack_trace };
    _ = try sema.analyzeCall(block, panic_fn, src, src, .auto, false, args);
    return always_noreturn;
}

fn safetyPanic(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    panic_id: PanicId,
) CompileError!Zir.Inst.Index {
    const msg = switch (panic_id) {
        .unreach => "reached unreachable code",
        .unwrap_null => "attempt to use null value",
        .unwrap_errunion => "unreachable error occurred",
        .cast_to_null => "cast causes pointer to be null",
        .incorrect_alignment => "incorrect alignment",
        .invalid_error_code => "invalid error code",
        .index_out_of_bounds => "attempt to index out of bounds",
        .cast_truncated_data => "integer cast truncated bits",
    };

    const msg_inst = msg_inst: {
        // TODO instead of making a new decl for every panic in the entire compilation,
        // introduce the concept of a reference-counted decl for these
        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();
        break :msg_inst try sema.analyzeDeclRef(try anon_decl.finish(
            try Type.Tag.array_u8.create(anon_decl.arena(), msg.len),
            try Value.Tag.bytes.create(anon_decl.arena(), msg),
            0, // default alignment
        ));
    };

    const casted_msg_inst = try sema.coerce(block, Type.initTag(.const_slice_u8), msg_inst, src);
    return sema.panicWithMsg(block, src, casted_msg_inst);
}

fn emitBackwardBranch(sema: *Sema, block: *Block, src: LazySrcLoc) !void {
    sema.branch_count += 1;
    if (sema.branch_count > sema.branch_quota) {
        // TODO show the "called from here" stack
        return sema.fail(block, src, "evaluation exceeded {d} backwards branches", .{sema.branch_quota});
    }
}

fn fieldVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    object: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    // When editing this function, note that there is corresponding logic to be edited
    // in `fieldPtr`. This function takes a value and returns a value.

    const arena = sema.arena;
    const object_src = src; // TODO better source location
    const object_ty = sema.typeOf(object);

    // Zig allows dereferencing a single pointer during field lookup. Note that
    // we don't actually need to generate the dereference some field lookups, like the
    // length of arrays and other comptime operations.
    const is_pointer_to = object_ty.isSinglePointer();

    const inner_ty = if (is_pointer_to)
        object_ty.childType()
    else
        object_ty;

    const target = sema.mod.getTarget();

    switch (inner_ty.zigTypeTag()) {
        .Array => {
            if (mem.eql(u8, field_name, "len")) {
                return sema.addConstant(
                    Type.usize,
                    try Value.Tag.int_u64.create(arena, inner_ty.arrayLen()),
                );
            } else {
                return sema.fail(
                    block,
                    field_name_src,
                    "no member named '{s}' in '{}'",
                    .{ field_name, object_ty.fmt(target) },
                );
            }
        },
        .Pointer => {
            const ptr_info = inner_ty.ptrInfo().data;
            if (ptr_info.size == .Slice) {
                if (mem.eql(u8, field_name, "ptr")) {
                    const slice = if (is_pointer_to)
                        try sema.analyzeLoad(block, src, object, object_src)
                    else
                        object;
                    return sema.analyzeSlicePtr(block, object_src, slice, inner_ty);
                } else if (mem.eql(u8, field_name, "len")) {
                    const slice = if (is_pointer_to)
                        try sema.analyzeLoad(block, src, object, object_src)
                    else
                        object;
                    return sema.analyzeSliceLen(block, src, slice);
                } else {
                    return sema.fail(
                        block,
                        field_name_src,
                        "no member named '{s}' in '{}'",
                        .{ field_name, object_ty.fmt(target) },
                    );
                }
            } else if (ptr_info.pointee_type.zigTypeTag() == .Array) {
                if (mem.eql(u8, field_name, "len")) {
                    return sema.addConstant(
                        Type.usize,
                        try Value.Tag.int_u64.create(arena, ptr_info.pointee_type.arrayLen()),
                    );
                } else {
                    return sema.fail(
                        block,
                        field_name_src,
                        "no member named '{s}' in '{}'",
                        .{ field_name, ptr_info.pointee_type.fmt(target) },
                    );
                }
            }
        },
        .Type => {
            const dereffed_type = if (is_pointer_to)
                try sema.analyzeLoad(block, src, object, object_src)
            else
                object;

            const val = (try sema.resolveDefinedValue(block, object_src, dereffed_type)).?;
            var to_type_buffer: Value.ToTypeBuffer = undefined;
            const child_type = val.toType(&to_type_buffer);

            switch (try child_type.zigTypeTagOrPoison()) {
                .ErrorSet => {
                    const name: []const u8 = if (child_type.castTag(.error_set)) |payload| blk: {
                        if (payload.data.names.getEntry(field_name)) |entry| {
                            break :blk entry.key_ptr.*;
                        }
                        return sema.fail(block, src, "no error named '{s}' in '{}'", .{
                            field_name, child_type.fmt(target),
                        });
                    } else (try sema.mod.getErrorValue(field_name)).key;

                    return sema.addConstant(
                        try child_type.copy(arena),
                        try Value.Tag.@"error".create(arena, .{ .name = name }),
                    );
                },
                .Union => {
                    const union_ty = try sema.resolveTypeFields(block, src, child_type);

                    if (union_ty.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupVal(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    if (union_ty.unionTagType()) |enum_ty| {
                        if (enum_ty.enumFieldIndex(field_name)) |field_index_usize| {
                            const field_index = @intCast(u32, field_index_usize);
                            return sema.addConstant(
                                enum_ty,
                                try Value.Tag.enum_field_index.create(sema.arena, field_index),
                            );
                        }
                    }
                    return sema.failWithBadMemberAccess(block, union_ty, field_name_src, field_name);
                },
                .Enum => {
                    if (child_type.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupVal(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    const field_index_usize = child_type.enumFieldIndex(field_name) orelse
                        return sema.failWithBadMemberAccess(block, child_type, field_name_src, field_name);
                    const field_index = @intCast(u32, field_index_usize);
                    const enum_val = try Value.Tag.enum_field_index.create(arena, field_index);
                    return sema.addConstant(try child_type.copy(arena), enum_val);
                },
                .Struct, .Opaque => {
                    if (child_type.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupVal(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    // TODO add note: declared here
                    const kw_name = switch (child_type.zigTypeTag()) {
                        .Struct => "struct",
                        .Opaque => "opaque",
                        .Union => "union",
                        else => unreachable,
                    };
                    return sema.fail(block, src, "{s} '{}' has no member named '{s}'", .{
                        kw_name, child_type.fmt(target), field_name,
                    });
                },
                else => return sema.fail(block, src, "type '{}' has no members", .{child_type.fmt(target)}),
            }
        },
        .Struct => if (is_pointer_to) {
            // Avoid loading the entire struct by fetching a pointer and loading that
            const field_ptr = try sema.structFieldPtr(block, src, object, field_name, field_name_src, inner_ty);
            return sema.analyzeLoad(block, src, field_ptr, object_src);
        } else {
            return sema.structFieldVal(block, src, object, field_name, field_name_src, inner_ty);
        },
        .Union => if (is_pointer_to) {
            // Avoid loading the entire union by fetching a pointer and loading that
            const field_ptr = try sema.unionFieldPtr(block, src, object, field_name, field_name_src, inner_ty);
            return sema.analyzeLoad(block, src, field_ptr, object_src);
        } else {
            return sema.unionFieldVal(block, src, object, field_name, field_name_src, inner_ty);
        },
        else => {},
    }
    return sema.fail(block, src, "type '{}' does not support field access", .{object_ty.fmt(target)});
}

fn fieldPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    object_ptr: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    // When editing this function, note that there is corresponding logic to be edited
    // in `fieldVal`. This function takes a pointer and returns a pointer.

    const target = sema.mod.getTarget();
    const object_ptr_src = src; // TODO better source location
    const object_ptr_ty = sema.typeOf(object_ptr);
    const object_ty = switch (object_ptr_ty.zigTypeTag()) {
        .Pointer => object_ptr_ty.elemType(),
        else => return sema.fail(block, object_ptr_src, "expected pointer, found '{}'", .{object_ptr_ty.fmt(target)}),
    };

    // Zig allows dereferencing a single pointer during field lookup. Note that
    // we don't actually need to generate the dereference some field lookups, like the
    // length of arrays and other comptime operations.
    const is_pointer_to = object_ty.isSinglePointer();

    const inner_ty = if (is_pointer_to)
        object_ty.childType()
    else
        object_ty;

    switch (inner_ty.zigTypeTag()) {
        .Array => {
            if (mem.eql(u8, field_name, "len")) {
                var anon_decl = try block.startAnonDecl(src);
                defer anon_decl.deinit();
                return sema.analyzeDeclRef(try anon_decl.finish(
                    Type.usize,
                    try Value.Tag.int_u64.create(anon_decl.arena(), inner_ty.arrayLen()),
                    0, // default alignment
                ));
            } else {
                return sema.fail(
                    block,
                    field_name_src,
                    "no member named '{s}' in '{}'",
                    .{ field_name, object_ty.fmt(target) },
                );
            }
        },
        .Pointer => if (inner_ty.isSlice()) {
            const inner_ptr = if (is_pointer_to)
                try sema.analyzeLoad(block, src, object_ptr, object_ptr_src)
            else
                object_ptr;

            if (mem.eql(u8, field_name, "ptr")) {
                const buf = try sema.arena.create(Type.SlicePtrFieldTypeBuffer);
                const slice_ptr_ty = inner_ty.slicePtrFieldType(buf);

                if (try sema.resolveDefinedValue(block, object_ptr_src, inner_ptr)) |val| {
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();

                    return sema.analyzeDeclRef(try anon_decl.finish(
                        try slice_ptr_ty.copy(anon_decl.arena()),
                        try val.slicePtr().copy(anon_decl.arena()),
                        0, // default alignment
                    ));
                }
                try sema.requireRuntimeBlock(block, src);

                const result_ty = try Type.ptr(sema.arena, target, .{
                    .pointee_type = slice_ptr_ty,
                    .mutable = object_ptr_ty.ptrIsMutable(),
                    .@"addrspace" = object_ptr_ty.ptrAddressSpace(),
                });

                return block.addTyOp(.ptr_slice_ptr_ptr, result_ty, inner_ptr);
            } else if (mem.eql(u8, field_name, "len")) {
                if (try sema.resolveDefinedValue(block, object_ptr_src, inner_ptr)) |val| {
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();

                    return sema.analyzeDeclRef(try anon_decl.finish(
                        Type.usize,
                        try Value.Tag.int_u64.create(anon_decl.arena(), val.sliceLen(target)),
                        0, // default alignment
                    ));
                }
                try sema.requireRuntimeBlock(block, src);

                const result_ty = try Type.ptr(sema.arena, target, .{
                    .pointee_type = Type.usize,
                    .mutable = object_ptr_ty.ptrIsMutable(),
                    .@"addrspace" = object_ptr_ty.ptrAddressSpace(),
                });

                return block.addTyOp(.ptr_slice_len_ptr, result_ty, inner_ptr);
            } else {
                return sema.fail(
                    block,
                    field_name_src,
                    "no member named '{s}' in '{}'",
                    .{ field_name, object_ty.fmt(target) },
                );
            }
        },
        .Type => {
            _ = try sema.resolveConstValue(block, object_ptr_src, object_ptr);
            const result = try sema.analyzeLoad(block, src, object_ptr, object_ptr_src);
            const inner = if (is_pointer_to)
                try sema.analyzeLoad(block, src, result, object_ptr_src)
            else
                result;

            const val = (sema.resolveDefinedValue(block, src, inner) catch unreachable).?;
            var to_type_buffer: Value.ToTypeBuffer = undefined;
            const child_type = val.toType(&to_type_buffer);

            switch (child_type.zigTypeTag()) {
                .ErrorSet => {
                    // TODO resolve inferred error sets
                    const name: []const u8 = if (child_type.castTag(.error_set)) |payload| blk: {
                        if (payload.data.names.getEntry(field_name)) |entry| {
                            break :blk entry.key_ptr.*;
                        }
                        return sema.fail(block, src, "no error named '{s}' in '{}'", .{
                            field_name, child_type.fmt(target),
                        });
                    } else (try sema.mod.getErrorValue(field_name)).key;

                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    return sema.analyzeDeclRef(try anon_decl.finish(
                        try child_type.copy(anon_decl.arena()),
                        try Value.Tag.@"error".create(anon_decl.arena(), .{ .name = name }),
                        0, // default alignment
                    ));
                },
                .Union => {
                    if (child_type.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupRef(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    if (child_type.unionTagType()) |enum_ty| {
                        if (enum_ty.enumFieldIndex(field_name)) |field_index| {
                            const field_index_u32 = @intCast(u32, field_index);
                            var anon_decl = try block.startAnonDecl(src);
                            defer anon_decl.deinit();
                            return sema.analyzeDeclRef(try anon_decl.finish(
                                try enum_ty.copy(anon_decl.arena()),
                                try Value.Tag.enum_field_index.create(anon_decl.arena(), field_index_u32),
                                0, // default alignment
                            ));
                        }
                    }
                    return sema.failWithBadMemberAccess(block, child_type, field_name_src, field_name);
                },
                .Enum => {
                    if (child_type.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupRef(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    const field_index = child_type.enumFieldIndex(field_name) orelse {
                        return sema.failWithBadMemberAccess(block, child_type, field_name_src, field_name);
                    };
                    const field_index_u32 = @intCast(u32, field_index);
                    var anon_decl = try block.startAnonDecl(src);
                    defer anon_decl.deinit();
                    return sema.analyzeDeclRef(try anon_decl.finish(
                        try child_type.copy(anon_decl.arena()),
                        try Value.Tag.enum_field_index.create(anon_decl.arena(), field_index_u32),
                        0, // default alignment
                    ));
                },
                .Struct, .Opaque => {
                    if (child_type.getNamespace()) |namespace| {
                        if (try sema.namespaceLookupRef(block, src, namespace, field_name)) |inst| {
                            return inst;
                        }
                    }
                    return sema.failWithBadMemberAccess(block, child_type, field_name_src, field_name);
                },
                else => return sema.fail(block, src, "type '{}' has no members", .{child_type.fmt(target)}),
            }
        },
        .Struct => {
            const inner_ptr = if (is_pointer_to)
                try sema.analyzeLoad(block, src, object_ptr, object_ptr_src)
            else
                object_ptr;
            return sema.structFieldPtr(block, src, inner_ptr, field_name, field_name_src, inner_ty);
        },
        .Union => {
            const inner_ptr = if (is_pointer_to)
                try sema.analyzeLoad(block, src, object_ptr, object_ptr_src)
            else
                object_ptr;
            return sema.unionFieldPtr(block, src, inner_ptr, field_name, field_name_src, inner_ty);
        },
        else => {},
    }
    return sema.fail(block, src, "type '{}' does not support field access (fieldPtr, {}.{s})", .{ object_ty.fmt(target), object_ptr_ty.fmt(target), field_name });
}

fn fieldCallBind(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    raw_ptr: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    // When editing this function, note that there is corresponding logic to be edited
    // in `fieldVal`. This function takes a pointer and returns a pointer.

    const target = sema.mod.getTarget();
    const raw_ptr_src = src; // TODO better source location
    const raw_ptr_ty = sema.typeOf(raw_ptr);
    const inner_ty = if (raw_ptr_ty.zigTypeTag() == .Pointer and raw_ptr_ty.ptrSize() == .One)
        raw_ptr_ty.childType()
    else
        return sema.fail(block, raw_ptr_src, "expected single pointer, found '{}'", .{raw_ptr_ty.fmt(target)});

    // Optionally dereference a second pointer to get the concrete type.
    const is_double_ptr = inner_ty.zigTypeTag() == .Pointer and inner_ty.ptrSize() == .One;
    const concrete_ty = if (is_double_ptr) inner_ty.childType() else inner_ty;
    const ptr_ty = if (is_double_ptr) inner_ty else raw_ptr_ty;
    const object_ptr = if (is_double_ptr)
        try sema.analyzeLoad(block, src, raw_ptr, src)
    else
        raw_ptr;

    const arena = sema.arena;
    find_field: {
        switch (concrete_ty.zigTypeTag()) {
            .Struct => {
                const struct_ty = try sema.resolveTypeFields(block, src, concrete_ty);
                const struct_obj = struct_ty.castTag(.@"struct").?.data;

                const field_index_usize = struct_obj.fields.getIndex(field_name) orelse
                    break :find_field;
                const field_index = @intCast(u32, field_index_usize);
                const field = struct_obj.fields.values()[field_index];

                return finishFieldCallBind(sema, block, src, ptr_ty, field.ty, field_index, object_ptr);
            },
            .Union => {
                const union_ty = try sema.resolveTypeFields(block, src, concrete_ty);
                const fields = union_ty.unionFields();
                const field_index_usize = fields.getIndex(field_name) orelse break :find_field;
                const field_index = @intCast(u32, field_index_usize);
                const field = fields.values()[field_index];

                return finishFieldCallBind(sema, block, src, ptr_ty, field.ty, field_index, object_ptr);
            },
            .Type => {
                const namespace = try sema.analyzeLoad(block, src, object_ptr, src);
                return sema.fieldVal(block, src, namespace, field_name, field_name_src);
            },
            else => {},
        }
    }

    // If we get here, we need to look for a decl in the struct type instead.
    switch (concrete_ty.zigTypeTag()) {
        .Struct, .Opaque, .Union, .Enum => {
            if (concrete_ty.getNamespace()) |namespace| {
                if (try sema.namespaceLookupRef(block, src, namespace, field_name)) |inst| {
                    const decl_val = try sema.analyzeLoad(block, src, inst, src);
                    const decl_type = sema.typeOf(decl_val);
                    if (decl_type.zigTypeTag() == .Fn and
                        decl_type.fnParamLen() >= 1)
                    {
                        const first_param_type = decl_type.fnParamType(0);
                        const first_param_tag = first_param_type.tag();
                        // zig fmt: off
                        if (first_param_tag == .var_args_param or
                            first_param_tag == .generic_poison or (
                                first_param_type.zigTypeTag() == .Pointer and
                                (first_param_type.ptrSize() == .One or
                                first_param_type.ptrSize() == .C) and
                                first_param_type.childType().eql(concrete_ty, target)))
                        {
                            // zig fmt: on
                            // TODO: bound fn calls on rvalues should probably
                            // generate a by-value argument somehow.
                            const ty = Type.Tag.bound_fn.init();
                            const value = try Value.Tag.bound_fn.create(arena, .{
                                .func_inst = decl_val,
                                .arg0_inst = object_ptr,
                            });
                            return sema.addConstant(ty, value);
                        } else if (first_param_type.eql(concrete_ty, target)) {
                            var deref = try sema.analyzeLoad(block, src, object_ptr, src);
                            const ty = Type.Tag.bound_fn.init();
                            const value = try Value.Tag.bound_fn.create(arena, .{
                                .func_inst = decl_val,
                                .arg0_inst = deref,
                            });
                            return sema.addConstant(ty, value);
                        }
                    }
                }
            }
        },
        else => {},
    }

    return sema.fail(block, src, "type '{}' has no field or member function named '{s}'", .{ concrete_ty.fmt(target), field_name });
}

fn finishFieldCallBind(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr_ty: Type,
    field_ty: Type,
    field_index: u32,
    object_ptr: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const arena = sema.arena;
    const target = sema.mod.getTarget();
    const ptr_field_ty = try Type.ptr(arena, target, .{
        .pointee_type = field_ty,
        .mutable = ptr_ty.ptrIsMutable(),
        .@"addrspace" = ptr_ty.ptrAddressSpace(),
    });

    if (try sema.resolveDefinedValue(block, src, object_ptr)) |struct_ptr_val| {
        const pointer = try sema.addConstant(
            ptr_field_ty,
            try Value.Tag.field_ptr.create(arena, .{
                .container_ptr = struct_ptr_val,
                .container_ty = ptr_ty.childType(),
                .field_index = field_index,
            }),
        );
        return sema.analyzeLoad(block, src, pointer, src);
    }

    try sema.requireRuntimeBlock(block, src);
    const ptr_inst = try block.addStructFieldPtr(object_ptr, field_index, ptr_field_ty);
    return sema.analyzeLoad(block, src, ptr_inst, src);
}

fn namespaceLookup(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    namespace: *Namespace,
    decl_name: []const u8,
) CompileError!?*Decl {
    const gpa = sema.gpa;
    if (try sema.lookupInNamespace(block, src, namespace, decl_name, true)) |decl| {
        if (!decl.is_pub and decl.getFileScope() != block.getFileScope()) {
            const msg = msg: {
                const msg = try sema.errMsg(block, src, "'{s}' is not marked 'pub'", .{
                    decl_name,
                });
                errdefer msg.destroy(gpa);
                try sema.mod.errNoteNonLazy(decl.srcLoc(), msg, "declared here", .{});
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        }
        return decl;
    }
    return null;
}

fn namespaceLookupRef(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    namespace: *Namespace,
    decl_name: []const u8,
) CompileError!?Air.Inst.Ref {
    const decl = (try sema.namespaceLookup(block, src, namespace, decl_name)) orelse return null;
    return try sema.analyzeDeclRef(decl);
}

fn namespaceLookupVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    namespace: *Namespace,
    decl_name: []const u8,
) CompileError!?Air.Inst.Ref {
    const decl = (try sema.namespaceLookup(block, src, namespace, decl_name)) orelse return null;
    return try sema.analyzeDeclVal(block, src, decl);
}

fn structFieldPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    struct_ptr: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
    unresolved_struct_ty: Type,
) CompileError!Air.Inst.Ref {
    assert(unresolved_struct_ty.zigTypeTag() == .Struct);

    const struct_ty = try sema.resolveTypeFields(block, src, unresolved_struct_ty);
    try sema.resolveStructLayout(block, src, struct_ty);

    if (struct_ty.isTuple()) {
        if (mem.eql(u8, field_name, "len")) {
            const len_inst = try sema.addIntUnsigned(Type.usize, struct_ty.structFieldCount());
            return sema.analyzeRef(block, src, len_inst);
        }
        const field_index = try sema.tupleFieldIndex(block, struct_ty, field_name, field_name_src);
        return sema.tupleFieldPtr(block, src, struct_ptr, field_name_src, field_index);
    } else if (struct_ty.isAnonStruct()) {
        const field_index = try sema.anonStructFieldIndex(block, struct_ty, field_name, field_name_src);
        return sema.tupleFieldPtr(block, src, struct_ptr, field_name_src, field_index);
    }

    const struct_obj = struct_ty.castTag(.@"struct").?.data;

    const field_index_big = struct_obj.fields.getIndex(field_name) orelse
        return sema.failWithBadStructFieldAccess(block, struct_obj, field_name_src, field_name);
    const field_index = @intCast(u32, field_index_big);

    return sema.structFieldPtrByIndex(block, src, struct_ptr, field_index, struct_obj, field_name_src);
}

fn structFieldPtrByIndex(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    struct_ptr: Air.Inst.Ref,
    field_index: u32,
    struct_obj: *Module.Struct,
    field_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const field = struct_obj.fields.values()[field_index];
    const struct_ptr_ty = sema.typeOf(struct_ptr);
    const struct_ptr_ty_info = struct_ptr_ty.ptrInfo().data;

    var ptr_ty_data: Type.Payload.Pointer.Data = .{
        .pointee_type = field.ty,
        .mutable = struct_ptr_ty_info.mutable,
        .@"addrspace" = struct_ptr_ty_info.@"addrspace",
    };

    const target = sema.mod.getTarget();

    // TODO handle when the struct pointer is overaligned, we should return a potentially
    // over-aligned field pointer too.
    if (struct_obj.layout == .Packed) {
        comptime assert(Type.packed_struct_layout_version == 2);

        var running_bits: u16 = 0;
        for (struct_obj.fields.values()) |f, i| {
            if (!(try sema.typeHasRuntimeBits(block, field_src, f.ty))) continue;

            if (i == field_index) {
                ptr_ty_data.bit_offset = running_bits;
            }
            running_bits += @intCast(u16, f.ty.bitSize(target));
        }
        ptr_ty_data.host_size = (running_bits + 7) / 8;

        // If this is a packed struct embedded in another one, we need to offset
        // the bits against each other.
        if (struct_ptr_ty_info.host_size != 0) {
            ptr_ty_data.host_size = struct_ptr_ty_info.host_size;
            ptr_ty_data.bit_offset += struct_ptr_ty_info.bit_offset;
        }
    } else {
        ptr_ty_data.@"align" = field.abi_align;
    }

    const ptr_field_ty = try Type.ptr(sema.arena, target, ptr_ty_data);

    if (field.is_comptime) {
        var anon_decl = try block.startAnonDecl(field_src);
        defer anon_decl.deinit();
        const decl = try anon_decl.finish(
            try field.ty.copy(anon_decl.arena()),
            try field.default_val.copy(anon_decl.arena()),
            ptr_ty_data.@"align",
        );
        return sema.analyzeDeclRef(decl);
    }

    if (try sema.resolveDefinedValue(block, src, struct_ptr)) |struct_ptr_val| {
        return sema.addConstant(
            ptr_field_ty,
            try Value.Tag.field_ptr.create(sema.arena, .{
                .container_ptr = struct_ptr_val,
                .container_ty = struct_ptr_ty.childType(),
                .field_index = field_index,
            }),
        );
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addStructFieldPtr(struct_ptr, field_index, ptr_field_ty);
}

fn structFieldVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    struct_byval: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
    unresolved_struct_ty: Type,
) CompileError!Air.Inst.Ref {
    assert(unresolved_struct_ty.zigTypeTag() == .Struct);

    const struct_ty = try sema.resolveTypeFields(block, src, unresolved_struct_ty);
    switch (struct_ty.tag()) {
        .tuple, .empty_struct_literal => return sema.tupleFieldVal(block, src, struct_byval, field_name, field_name_src, struct_ty),
        .anon_struct => {
            const field_index = try sema.anonStructFieldIndex(block, struct_ty, field_name, field_name_src);
            return tupleFieldValByIndex(sema, block, src, struct_byval, field_index, struct_ty);
        },
        .@"struct" => {
            const struct_obj = struct_ty.castTag(.@"struct").?.data;

            const field_index_usize = struct_obj.fields.getIndex(field_name) orelse
                return sema.failWithBadStructFieldAccess(block, struct_obj, field_name_src, field_name);
            const field_index = @intCast(u32, field_index_usize);
            const field = struct_obj.fields.values()[field_index];

            if (field.is_comptime) {
                return sema.addConstant(field.ty, field.default_val);
            }

            if (try sema.resolveMaybeUndefVal(block, src, struct_byval)) |struct_val| {
                if (struct_val.isUndef()) return sema.addConstUndef(field.ty);
                if ((try sema.typeHasOnePossibleValue(block, src, field.ty))) |opv| {
                    return sema.addConstant(field.ty, opv);
                }

                const field_values = struct_val.castTag(.aggregate).?.data;
                return sema.addConstant(field.ty, field_values[field_index]);
            }

            try sema.requireRuntimeBlock(block, src);
            return block.addStructFieldVal(struct_byval, field_index, field.ty);
        },
        else => unreachable,
    }
}

fn tupleFieldVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    tuple_byval: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
    tuple_ty: Type,
) CompileError!Air.Inst.Ref {
    if (mem.eql(u8, field_name, "len")) {
        return sema.addIntUnsigned(Type.usize, tuple_ty.structFieldCount());
    }
    const field_index = try sema.tupleFieldIndex(block, tuple_ty, field_name, field_name_src);
    return tupleFieldValByIndex(sema, block, src, tuple_byval, field_index, tuple_ty);
}

/// Don't forget to check for "len" before calling this.
fn tupleFieldIndex(
    sema: *Sema,
    block: *Block,
    tuple_ty: Type,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
) CompileError!u32 {
    const target = sema.mod.getTarget();
    const field_index = std.fmt.parseUnsigned(u32, field_name, 10) catch |err| {
        return sema.fail(block, field_name_src, "tuple {} has no such field '{s}': {s}", .{
            tuple_ty.fmt(target), field_name, @errorName(err),
        });
    };
    if (field_index >= tuple_ty.structFieldCount()) {
        return sema.fail(block, field_name_src, "tuple {} has no such field '{s}'", .{
            tuple_ty.fmt(target), field_name,
        });
    }
    return field_index;
}

fn tupleFieldValByIndex(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    tuple_byval: Air.Inst.Ref,
    field_index: u32,
    tuple_ty: Type,
) CompileError!Air.Inst.Ref {
    const tuple = tuple_ty.tupleFields();
    const field_ty = tuple.types[field_index];

    if (tuple.values[field_index].tag() != .unreachable_value) {
        return sema.addConstant(field_ty, tuple.values[field_index]);
    }

    if (try sema.resolveMaybeUndefVal(block, src, tuple_byval)) |tuple_val| {
        if (tuple_val.isUndef()) return sema.addConstUndef(field_ty);
        if ((try sema.typeHasOnePossibleValue(block, src, field_ty))) |opv| {
            return sema.addConstant(field_ty, opv);
        }
        const field_values = tuple_val.castTag(.aggregate).?.data;
        return sema.addConstant(field_ty, field_values[field_index]);
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addStructFieldVal(tuple_byval, field_index, field_ty);
}

fn unionFieldPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    union_ptr: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
    unresolved_union_ty: Type,
) CompileError!Air.Inst.Ref {
    const arena = sema.arena;
    assert(unresolved_union_ty.zigTypeTag() == .Union);

    const union_ptr_ty = sema.typeOf(union_ptr);
    const union_ty = try sema.resolveTypeFields(block, src, unresolved_union_ty);
    const union_obj = union_ty.cast(Type.Payload.Union).?.data;
    const field_index = try sema.unionFieldIndex(block, union_ty, field_name, field_name_src);
    const field = union_obj.fields.values()[field_index];
    const target = sema.mod.getTarget();
    const ptr_field_ty = try Type.ptr(arena, target, .{
        .pointee_type = field.ty,
        .mutable = union_ptr_ty.ptrIsMutable(),
        .@"addrspace" = union_ptr_ty.ptrAddressSpace(),
    });

    if (try sema.resolveDefinedValue(block, src, union_ptr)) |union_ptr_val| {
        switch (union_obj.layout) {
            .Auto => {
                // TODO emit the access of inactive union field error commented out below.
                // In order to do that, we need to first solve the problem that AstGen
                // emits field_ptr instructions in order to initialize union values.
                // In such case we need to know that the field_ptr instruction (which is
                // calling this unionFieldPtr function) is *initializing* the union,
                // in which case we would skip this check, and in fact we would actually
                // set the union tag here and the payload to undefined.

                //const tag_and_val = union_val.castTag(.@"union").?.data;
                //var field_tag_buf: Value.Payload.U32 = .{
                //    .base = .{ .tag = .enum_field_index },
                //    .data = field_index,
                //};
                //const field_tag = Value.initPayload(&field_tag_buf.base);
                //const tag_matches = tag_and_val.tag.eql(field_tag, union_obj.tag_ty, target);
                //if (!tag_matches) {
                //    // TODO enhance this saying which one was active
                //    // and which one was accessed, and showing where the union was declared.
                //    return sema.fail(block, src, "access of inactive union field", .{});
                //}
                // TODO add runtime safety check for the active tag
            },
            .Packed, .Extern => {},
        }
        return sema.addConstant(
            ptr_field_ty,
            try Value.Tag.field_ptr.create(arena, .{
                .container_ptr = union_ptr_val,
                .container_ty = union_ty,
                .field_index = field_index,
            }),
        );
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addStructFieldPtr(union_ptr, field_index, ptr_field_ty);
}

fn unionFieldVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    union_byval: Air.Inst.Ref,
    field_name: []const u8,
    field_name_src: LazySrcLoc,
    unresolved_union_ty: Type,
) CompileError!Air.Inst.Ref {
    assert(unresolved_union_ty.zigTypeTag() == .Union);

    const union_ty = try sema.resolveTypeFields(block, src, unresolved_union_ty);
    const union_obj = union_ty.cast(Type.Payload.Union).?.data;
    const field_index = try sema.unionFieldIndex(block, union_ty, field_name, field_name_src);
    const field = union_obj.fields.values()[field_index];

    if (try sema.resolveMaybeUndefVal(block, src, union_byval)) |union_val| {
        if (union_val.isUndef()) return sema.addConstUndef(field.ty);

        const tag_and_val = union_val.castTag(.@"union").?.data;
        var field_tag_buf: Value.Payload.U32 = .{
            .base = .{ .tag = .enum_field_index },
            .data = field_index,
        };
        const field_tag = Value.initPayload(&field_tag_buf.base);
        const target = sema.mod.getTarget();
        const tag_matches = tag_and_val.tag.eql(field_tag, union_obj.tag_ty, target);
        switch (union_obj.layout) {
            .Auto => {
                if (tag_matches) {
                    return sema.addConstant(field.ty, tag_and_val.val);
                } else {
                    // TODO enhance this saying which one was active
                    // and which one was accessed, and showing where the union was declared.
                    return sema.fail(block, src, "access of inactive union field", .{});
                }
            },
            .Packed, .Extern => {
                if (tag_matches) {
                    return sema.addConstant(field.ty, tag_and_val.val);
                } else {
                    const old_ty = union_ty.unionFieldType(tag_and_val.tag, target);
                    const new_val = try sema.bitCastVal(block, src, tag_and_val.val, old_ty, field.ty, 0);
                    return sema.addConstant(field.ty, new_val);
                }
            },
        }
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addStructFieldVal(union_byval, field_index, field.ty);
}

fn elemPtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    indexable_ptr: Air.Inst.Ref,
    elem_index: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const indexable_ptr_src = src; // TODO better source location
    const indexable_ptr_ty = sema.typeOf(indexable_ptr);
    const target = sema.mod.getTarget();
    const indexable_ty = switch (indexable_ptr_ty.zigTypeTag()) {
        .Pointer => indexable_ptr_ty.elemType(),
        else => return sema.fail(block, indexable_ptr_src, "expected pointer, found '{}'", .{indexable_ptr_ty.fmt(target)}),
    };
    if (!indexable_ty.isIndexable()) {
        return sema.fail(block, src, "element access of non-indexable type '{}'", .{indexable_ty.fmt(target)});
    }

    switch (indexable_ty.zigTypeTag()) {
        .Pointer => {
            // In all below cases, we have to deref the ptr operand to get the actual indexable pointer.
            const indexable = try sema.analyzeLoad(block, indexable_ptr_src, indexable_ptr, indexable_ptr_src);
            const result_ty = try indexable_ty.elemPtrType(sema.arena, target);
            switch (indexable_ty.ptrSize()) {
                .Slice => return sema.elemPtrSlice(block, indexable_ptr_src, indexable, elem_index_src, elem_index),
                .Many, .C => {
                    const maybe_ptr_val = try sema.resolveDefinedValue(block, indexable_ptr_src, indexable);
                    const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);

                    const runtime_src = rs: {
                        const ptr_val = maybe_ptr_val orelse break :rs indexable_ptr_src;
                        const index_val = maybe_index_val orelse break :rs elem_index_src;
                        const index = @intCast(usize, index_val.toUnsignedInt(target));
                        const elem_ptr = try ptr_val.elemPtr(indexable_ty, sema.arena, index, target);
                        return sema.addConstant(result_ty, elem_ptr);
                    };

                    try sema.requireRuntimeBlock(block, runtime_src);
                    return block.addPtrElemPtr(indexable, elem_index, result_ty);
                },
                .One => {
                    assert(indexable_ty.childType().zigTypeTag() == .Array); // Guaranteed by isIndexable
                    return sema.elemPtrArray(block, indexable_ptr_src, indexable, elem_index_src, elem_index);
                },
            }
        },
        .Array, .Vector => return sema.elemPtrArray(block, indexable_ptr_src, indexable_ptr, elem_index_src, elem_index),
        .Struct => {
            // Tuple field access.
            const index_val = try sema.resolveConstValue(block, elem_index_src, elem_index);
            const index = @intCast(u32, index_val.toUnsignedInt(target));
            return sema.tupleFieldPtr(block, src, indexable_ptr, elem_index_src, index);
        },
        else => unreachable,
    }
}

fn elemVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    indexable: Air.Inst.Ref,
    elem_index_uncasted: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const indexable_src = src; // TODO better source location
    const indexable_ty = sema.typeOf(indexable);
    const target = sema.mod.getTarget();

    if (!indexable_ty.isIndexable()) {
        return sema.fail(block, src, "element access of non-indexable type '{}'", .{indexable_ty.fmt(target)});
    }

    // TODO in case of a vector of pointers, we need to detect whether the element
    // index is a scalar or vector instead of unconditionally casting to usize.
    const elem_index = try sema.coerce(block, Type.usize, elem_index_uncasted, elem_index_src);

    switch (indexable_ty.zigTypeTag()) {
        .Pointer => switch (indexable_ty.ptrSize()) {
            .Slice => return sema.elemValSlice(block, indexable_src, indexable, elem_index_src, elem_index),
            .Many, .C => {
                const maybe_indexable_val = try sema.resolveDefinedValue(block, indexable_src, indexable);
                const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);

                const runtime_src = rs: {
                    const indexable_val = maybe_indexable_val orelse break :rs indexable_src;
                    const index_val = maybe_index_val orelse break :rs elem_index_src;
                    const index = @intCast(usize, index_val.toUnsignedInt(target));
                    const elem_ptr_val = try indexable_val.elemPtr(indexable_ty, sema.arena, index, target);
                    if (try sema.pointerDeref(block, indexable_src, elem_ptr_val, indexable_ty)) |elem_val| {
                        return sema.addConstant(indexable_ty.elemType2(), elem_val);
                    }
                    break :rs indexable_src;
                };

                try sema.requireRuntimeBlock(block, runtime_src);
                return block.addBinOp(.ptr_elem_val, indexable, elem_index);
            },
            .One => {
                assert(indexable_ty.childType().zigTypeTag() == .Array); // Guaranteed by isIndexable
                const elem_ptr = try sema.elemPtr(block, indexable_src, indexable, elem_index, elem_index_src);
                return sema.analyzeLoad(block, indexable_src, elem_ptr, elem_index_src);
            },
        },
        .Array => return elemValArray(sema, block, indexable_src, indexable, elem_index_src, elem_index),
        .Vector => {
            // TODO: If the index is a vector, the result should be a vector.
            return elemValArray(sema, block, indexable_src, indexable, elem_index_src, elem_index);
        },
        .Struct => {
            // Tuple field access.
            const index_val = try sema.resolveConstValue(block, elem_index_src, elem_index);
            const index = @intCast(u32, index_val.toUnsignedInt(target));
            return tupleField(sema, block, indexable_src, indexable, elem_index_src, index);
        },
        else => unreachable,
    }
}

fn tupleFieldPtr(
    sema: *Sema,
    block: *Block,
    tuple_ptr_src: LazySrcLoc,
    tuple_ptr: Air.Inst.Ref,
    field_index_src: LazySrcLoc,
    field_index: u32,
) CompileError!Air.Inst.Ref {
    const tuple_ptr_ty = sema.typeOf(tuple_ptr);
    const tuple_ty = tuple_ptr_ty.childType();
    const tuple_fields = tuple_ty.tupleFields();

    if (tuple_fields.types.len == 0) {
        return sema.fail(block, field_index_src, "indexing into empty tuple", .{});
    }

    if (field_index >= tuple_fields.types.len) {
        return sema.fail(block, field_index_src, "index {d} outside tuple of length {d}", .{
            field_index, tuple_fields.types.len,
        });
    }

    const field_ty = tuple_fields.types[field_index];
    const target = sema.mod.getTarget();
    const ptr_field_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = field_ty,
        .mutable = tuple_ptr_ty.ptrIsMutable(),
        .@"addrspace" = tuple_ptr_ty.ptrAddressSpace(),
    });

    if (try sema.resolveMaybeUndefVal(block, tuple_ptr_src, tuple_ptr)) |tuple_ptr_val| {
        return sema.addConstant(
            ptr_field_ty,
            try Value.Tag.field_ptr.create(sema.arena, .{
                .container_ptr = tuple_ptr_val,
                .container_ty = tuple_ty,
                .field_index = field_index,
            }),
        );
    }

    try sema.requireRuntimeBlock(block, tuple_ptr_src);
    return block.addStructFieldPtr(tuple_ptr, field_index, ptr_field_ty);
}

fn tupleField(
    sema: *Sema,
    block: *Block,
    tuple_src: LazySrcLoc,
    tuple: Air.Inst.Ref,
    field_index_src: LazySrcLoc,
    field_index: u32,
) CompileError!Air.Inst.Ref {
    const tuple_ty = sema.typeOf(tuple);
    const tuple_fields = tuple_ty.tupleFields();

    if (tuple_fields.types.len == 0) {
        return sema.fail(block, field_index_src, "indexing into empty tuple", .{});
    }

    if (field_index >= tuple_fields.types.len) {
        return sema.fail(block, field_index_src, "index {d} outside tuple of length {d}", .{
            field_index, tuple_fields.types.len,
        });
    }

    const field_ty = tuple_fields.types[field_index];
    const field_val = tuple_fields.values[field_index];

    if (field_val.tag() != .unreachable_value) {
        return sema.addConstant(field_ty, field_val); // comptime field
    }

    if (try sema.resolveMaybeUndefVal(block, tuple_src, tuple)) |tuple_val| {
        if (tuple_val.isUndef()) return sema.addConstUndef(field_ty);
        const field_values = tuple_val.castTag(.aggregate).?.data;
        return sema.addConstant(field_ty, field_values[field_index]);
    }

    try sema.requireRuntimeBlock(block, tuple_src);
    return block.addStructFieldVal(tuple, field_index, field_ty);
}

fn elemValArray(
    sema: *Sema,
    block: *Block,
    array_src: LazySrcLoc,
    array: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
    elem_index: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const array_ty = sema.typeOf(array);
    const array_sent = array_ty.sentinel() != null;
    const array_len = array_ty.arrayLen();
    const array_len_s = array_len + @boolToInt(array_sent);
    const elem_ty = array_ty.childType();

    if (array_len_s == 0) {
        return sema.fail(block, elem_index_src, "indexing into empty array", .{});
    }

    const maybe_undef_array_val = try sema.resolveMaybeUndefVal(block, array_src, array);
    // index must be defined since it can access out of bounds
    const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);
    const target = sema.mod.getTarget();

    if (maybe_index_val) |index_val| {
        const index = @intCast(usize, index_val.toUnsignedInt(target));
        if (index >= array_len_s) {
            const sentinel_label: []const u8 = if (array_sent) " +1 (sentinel)" else "";
            return sema.fail(block, elem_index_src, "index {d} outside array of length {d}{s}", .{ index, array_len, sentinel_label });
        }
    }
    if (maybe_undef_array_val) |array_val| {
        if (array_val.isUndef()) {
            return sema.addConstUndef(elem_ty);
        }
        if (maybe_index_val) |index_val| {
            const index = @intCast(usize, index_val.toUnsignedInt(target));
            const elem_val = try array_val.elemValue(sema.arena, index);
            return sema.addConstant(elem_ty, elem_val);
        }
    }

    const runtime_src = if (maybe_undef_array_val != null) elem_index_src else array_src;
    try sema.requireRuntimeBlock(block, runtime_src);
    if (block.wantSafety()) {
        // Runtime check is only needed if unable to comptime check
        if (maybe_index_val == null) {
            const len_inst = try sema.addIntUnsigned(Type.usize, array_len);
            const cmp_op: Air.Inst.Tag = if (array_sent) .cmp_lte else .cmp_lt;
            const is_in_bounds = try block.addBinOp(cmp_op, elem_index, len_inst);
            try sema.addSafetyCheck(block, is_in_bounds, .index_out_of_bounds);
        }
    }
    return block.addBinOp(.array_elem_val, array, elem_index);
}

fn elemPtrArray(
    sema: *Sema,
    block: *Block,
    array_ptr_src: LazySrcLoc,
    array_ptr: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
    elem_index: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const target = sema.mod.getTarget();
    const array_ptr_ty = sema.typeOf(array_ptr);
    const array_ty = array_ptr_ty.childType();
    const array_sent = array_ty.sentinel() != null;
    const array_len = array_ty.arrayLen();
    const array_len_s = array_len + @boolToInt(array_sent);
    const elem_ptr_ty = try array_ptr_ty.elemPtrType(sema.arena, target);

    if (array_len_s == 0) {
        return sema.fail(block, elem_index_src, "indexing into empty array", .{});
    }

    const maybe_undef_array_ptr_val = try sema.resolveMaybeUndefVal(block, array_ptr_src, array_ptr);
    // index must be defined since it can index out of bounds
    const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);

    if (maybe_index_val) |index_val| {
        const index = @intCast(usize, index_val.toUnsignedInt(target));
        if (index >= array_len_s) {
            const sentinel_label: []const u8 = if (array_sent) " +1 (sentinel)" else "";
            return sema.fail(block, elem_index_src, "index {d} outside array of length {d}{s}", .{ index, array_len, sentinel_label });
        }
    }
    if (maybe_undef_array_ptr_val) |array_ptr_val| {
        if (array_ptr_val.isUndef()) {
            return sema.addConstUndef(elem_ptr_ty);
        }
        if (maybe_index_val) |index_val| {
            const index = @intCast(usize, index_val.toUnsignedInt(target));
            const elem_ptr = try array_ptr_val.elemPtr(array_ptr_ty, sema.arena, index, target);
            return sema.addConstant(elem_ptr_ty, elem_ptr);
        }
    }

    const runtime_src = if (maybe_undef_array_ptr_val != null) elem_index_src else array_ptr_src;
    try sema.requireRuntimeBlock(block, runtime_src);
    if (block.wantSafety()) {
        // Runtime check is only needed if unable to comptime check
        if (maybe_index_val == null) {
            const len_inst = try sema.addIntUnsigned(Type.usize, array_len);
            const cmp_op: Air.Inst.Tag = if (array_sent) .cmp_lte else .cmp_lt;
            const is_in_bounds = try block.addBinOp(cmp_op, elem_index, len_inst);
            try sema.addSafetyCheck(block, is_in_bounds, .index_out_of_bounds);
        }
    }
    return block.addPtrElemPtr(array_ptr, elem_index, elem_ptr_ty);
}

fn elemValSlice(
    sema: *Sema,
    block: *Block,
    slice_src: LazySrcLoc,
    slice: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
    elem_index: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const slice_ty = sema.typeOf(slice);
    const slice_sent = slice_ty.sentinel() != null;
    const elem_ty = slice_ty.elemType2();
    var runtime_src = slice_src;

    // slice must be defined since it can dereferenced as null
    const maybe_slice_val = try sema.resolveDefinedValue(block, slice_src, slice);
    // index must be defined since it can index out of bounds
    const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);
    const target = sema.mod.getTarget();

    if (maybe_slice_val) |slice_val| {
        runtime_src = elem_index_src;
        const slice_len = slice_val.sliceLen(target);
        const slice_len_s = slice_len + @boolToInt(slice_sent);
        if (slice_len_s == 0) {
            return sema.fail(block, elem_index_src, "indexing into empty slice", .{});
        }
        if (maybe_index_val) |index_val| {
            const index = @intCast(usize, index_val.toUnsignedInt(target));
            if (index >= slice_len_s) {
                const sentinel_label: []const u8 = if (slice_sent) " +1 (sentinel)" else "";
                return sema.fail(block, elem_index_src, "index {d} outside slice of length {d}{s}", .{ index, slice_len, sentinel_label });
            }
            const elem_ptr_val = try slice_val.elemPtr(slice_ty, sema.arena, index, target);
            if (try sema.pointerDeref(block, slice_src, elem_ptr_val, slice_ty)) |elem_val| {
                return sema.addConstant(elem_ty, elem_val);
            }
            runtime_src = slice_src;
        }
    }

    try sema.requireRuntimeBlock(block, runtime_src);
    if (block.wantSafety()) {
        const len_inst = if (maybe_slice_val) |slice_val|
            try sema.addIntUnsigned(Type.usize, slice_val.sliceLen(target))
        else
            try block.addTyOp(.slice_len, Type.usize, slice);
        const cmp_op: Air.Inst.Tag = if (slice_sent) .cmp_lte else .cmp_lt;
        const is_in_bounds = try block.addBinOp(cmp_op, elem_index, len_inst);
        try sema.addSafetyCheck(block, is_in_bounds, .index_out_of_bounds);
    }
    return block.addBinOp(.slice_elem_val, slice, elem_index);
}

fn elemPtrSlice(
    sema: *Sema,
    block: *Block,
    slice_src: LazySrcLoc,
    slice: Air.Inst.Ref,
    elem_index_src: LazySrcLoc,
    elem_index: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const target = sema.mod.getTarget();
    const slice_ty = sema.typeOf(slice);
    const slice_sent = slice_ty.sentinel() != null;
    const elem_ptr_ty = try slice_ty.elemPtrType(sema.arena, target);

    const maybe_undef_slice_val = try sema.resolveMaybeUndefVal(block, slice_src, slice);
    // index must be defined since it can index out of bounds
    const maybe_index_val = try sema.resolveDefinedValue(block, elem_index_src, elem_index);

    if (maybe_undef_slice_val) |slice_val| {
        if (slice_val.isUndef()) {
            return sema.addConstUndef(elem_ptr_ty);
        }
        const slice_len = slice_val.sliceLen(target);
        const slice_len_s = slice_len + @boolToInt(slice_sent);
        if (slice_len_s == 0) {
            return sema.fail(block, elem_index_src, "indexing into empty slice", .{});
        }
        if (maybe_index_val) |index_val| {
            const index = @intCast(usize, index_val.toUnsignedInt(target));
            if (index >= slice_len_s) {
                const sentinel_label: []const u8 = if (slice_sent) " +1 (sentinel)" else "";
                return sema.fail(block, elem_index_src, "index {d} outside slice of length {d}{s}", .{ index, slice_len, sentinel_label });
            }
            const elem_ptr_val = try slice_val.elemPtr(slice_ty, sema.arena, index, target);
            return sema.addConstant(elem_ptr_ty, elem_ptr_val);
        }
    }

    const runtime_src = if (maybe_undef_slice_val != null) elem_index_src else slice_src;
    try sema.requireRuntimeBlock(block, runtime_src);
    if (block.wantSafety()) {
        const len_inst = len: {
            if (maybe_undef_slice_val) |slice_val|
                if (!slice_val.isUndef())
                    break :len try sema.addIntUnsigned(Type.usize, slice_val.sliceLen(target));
            break :len try block.addTyOp(.slice_len, Type.usize, slice);
        };
        const cmp_op: Air.Inst.Tag = if (slice_sent) .cmp_lte else .cmp_lt;
        const is_in_bounds = try block.addBinOp(cmp_op, elem_index, len_inst);
        try sema.addSafetyCheck(block, is_in_bounds, .index_out_of_bounds);
    }
    return block.addSliceElemPtr(slice, elem_index, elem_ptr_ty);
}

fn coerce(
    sema: *Sema,
    block: *Block,
    dest_ty_unresolved: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    switch (dest_ty_unresolved.tag()) {
        .var_args_param => return sema.coerceVarArgParam(block, inst, inst_src),
        .generic_poison => return inst,
        else => {},
    }
    const dest_ty_src = inst_src; // TODO better source location
    const dest_ty = try sema.resolveTypeFields(block, dest_ty_src, dest_ty_unresolved);
    const inst_ty = try sema.resolveTypeFields(block, inst_src, sema.typeOf(inst));
    const target = sema.mod.getTarget();
    // If the types are the same, we can return the operand.
    if (dest_ty.eql(inst_ty, target))
        return inst;

    const arena = sema.arena;
    const maybe_inst_val = try sema.resolveMaybeUndefVal(block, inst_src, inst);

    const in_memory_result = try sema.coerceInMemoryAllowed(block, dest_ty, inst_ty, false, target, dest_ty_src, inst_src);
    if (in_memory_result == .ok) {
        if (maybe_inst_val) |val| {
            // Keep the comptime Value representation; take the new type.
            return sema.addConstant(dest_ty, val);
        }
        try sema.requireRuntimeBlock(block, inst_src);
        return block.addBitCast(dest_ty, inst);
    }

    const is_undef = if (maybe_inst_val) |val| val.isUndef() else false;

    switch (dest_ty.zigTypeTag()) {
        .Optional => {
            // undefined sets the optional bit also to undefined.
            if (is_undef) {
                return sema.addConstUndef(dest_ty);
            }

            // null to ?T
            if (inst_ty.zigTypeTag() == .Null) {
                return sema.addConstant(dest_ty, Value.@"null");
            }

            // cast from ?*T and ?[*]T to ?*anyopaque
            // but don't do it if the source type is a double pointer
            if (dest_ty.isPtrLikeOptional() and dest_ty.elemType2().tag() == .anyopaque and
                inst_ty.isPtrLikeOptional() and inst_ty.elemType2().zigTypeTag() != .Pointer)
            {
                return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
            }

            // T to ?T
            const child_type = try dest_ty.optionalChildAlloc(sema.arena);
            const intermediate = try sema.coerce(block, child_type, inst, inst_src);
            return sema.wrapOptional(block, dest_ty, intermediate, inst_src);
        },
        .Pointer => {
            const dest_info = dest_ty.ptrInfo().data;

            // Function body to function pointer.
            if (inst_ty.zigTypeTag() == .Fn) {
                const fn_val = try sema.resolveConstValue(block, inst_src, inst);
                const fn_decl = fn_val.castTag(.function).?.data.owner_decl;
                const inst_as_ptr = try sema.analyzeDeclRef(fn_decl);
                return sema.coerce(block, dest_ty, inst_as_ptr, inst_src);
            }

            // *T to *[1]T
            single_item: {
                if (dest_info.size != .One) break :single_item;
                if (!inst_ty.isSinglePointer()) break :single_item;
                const ptr_elem_ty = inst_ty.childType();
                const array_ty = dest_info.pointee_type;
                if (array_ty.zigTypeTag() != .Array) break :single_item;
                const array_elem_ty = array_ty.childType();
                const dest_is_mut = dest_info.mutable;
                if (inst_ty.isConstPtr() and dest_is_mut) break :single_item;
                if (inst_ty.isVolatilePtr() and !dest_info.@"volatile") break :single_item;
                if (inst_ty.ptrAddressSpace() != dest_info.@"addrspace") break :single_item;
                switch (try sema.coerceInMemoryAllowed(block, array_elem_ty, ptr_elem_ty, dest_is_mut, target, dest_ty_src, inst_src)) {
                    .ok => {},
                    .no_match => break :single_item,
                }
                return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
            }

            // Coercions where the source is a single pointer to an array.
            src_array_ptr: {
                if (!inst_ty.isSinglePointer()) break :src_array_ptr;
                const array_ty = inst_ty.childType();
                if (array_ty.zigTypeTag() != .Array) break :src_array_ptr;
                const len0 = array_ty.arrayLen() == 0;
                // We resolve here so that the backend has the layout of the elem type.
                const array_elem_type = try sema.resolveTypeFields(block, inst_src, array_ty.childType());
                const dest_is_mut = dest_info.mutable;
                if (inst_ty.isConstPtr() and dest_is_mut and !len0) break :src_array_ptr;
                if (inst_ty.isVolatilePtr() and !dest_info.@"volatile") break :src_array_ptr;
                if (inst_ty.ptrAddressSpace() != dest_info.@"addrspace") break :src_array_ptr;

                const dst_elem_type = dest_info.pointee_type;
                switch (try sema.coerceInMemoryAllowed(block, dst_elem_type, array_elem_type, dest_is_mut, target, dest_ty_src, inst_src)) {
                    .ok => {},
                    .no_match => break :src_array_ptr,
                }

                switch (dest_info.size) {
                    .Slice => {
                        // *[N]T to []T
                        return sema.coerceArrayPtrToSlice(block, dest_ty, inst, inst_src);
                    },
                    .C => {
                        // *[N]T to [*c]T
                        return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
                    },
                    .Many => {
                        // *[N]T to [*]T
                        // *[N:s]T to [*:s]T
                        // *[N:s]T to [*]T
                        if (dest_info.sentinel) |dst_sentinel| {
                            if (array_ty.sentinel()) |src_sentinel| {
                                if (src_sentinel.eql(dst_sentinel, dst_elem_type, target)) {
                                    return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
                                }
                            }
                        } else {
                            return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
                        }
                    },
                    .One => {},
                }
            }

            // coercion from C pointer
            if (inst_ty.isCPtr()) src_c_ptr: {
                // In this case we must add a safety check because the C pointer
                // could be null.
                const src_elem_ty = inst_ty.childType();
                const dest_is_mut = dest_info.mutable;
                const dst_elem_type = dest_info.pointee_type;
                switch (try sema.coerceInMemoryAllowed(block, dst_elem_type, src_elem_ty, dest_is_mut, target, dest_ty_src, inst_src)) {
                    .ok => {},
                    .no_match => break :src_c_ptr,
                }
                // TODO add safety check for null pointer
                return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
            }

            // cast from *T and [*]T to *anyopaque
            // but don't do it if the source type is a double pointer
            if (dest_info.pointee_type.tag() == .anyopaque and inst_ty.zigTypeTag() == .Pointer and
                inst_ty.childType().zigTypeTag() != .Pointer)
            {
                return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
            }

            switch (dest_info.size) {
                // coercion to C pointer
                .C => switch (inst_ty.zigTypeTag()) {
                    .Null => {
                        return sema.addConstant(dest_ty, Value.@"null");
                    },
                    .ComptimeInt => {
                        const addr = try sema.coerce(block, Type.usize, inst, inst_src);
                        return sema.coerceCompatiblePtrs(block, dest_ty, addr, inst_src);
                    },
                    .Int => {
                        const ptr_size_ty = switch (inst_ty.intInfo(target).signedness) {
                            .signed => Type.isize,
                            .unsigned => Type.usize,
                        };
                        const addr = try sema.coerce(block, ptr_size_ty, inst, inst_src);
                        return sema.coerceCompatiblePtrs(block, dest_ty, addr, inst_src);
                    },
                    .Pointer => p: {
                        const inst_info = inst_ty.ptrInfo().data;
                        switch (try sema.coerceInMemoryAllowed(
                            block,
                            dest_info.pointee_type,
                            inst_info.pointee_type,
                            dest_info.mutable,
                            target,
                            dest_ty_src,
                            inst_src,
                        )) {
                            .ok => {},
                            .no_match => break :p,
                        }
                        if (inst_info.size == .Slice) {
                            if (dest_info.sentinel == null or inst_info.sentinel == null or
                                !dest_info.sentinel.?.eql(inst_info.sentinel.?, dest_info.pointee_type, target))
                                break :p;

                            const slice_ptr = try sema.analyzeSlicePtr(block, inst_src, inst, inst_ty);
                            return sema.coerceCompatiblePtrs(block, dest_ty, slice_ptr, inst_src);
                        }
                        return sema.coerceCompatiblePtrs(block, dest_ty, inst, inst_src);
                    },
                    else => {},
                },
                .One => switch (dest_info.pointee_type.zigTypeTag()) {
                    .Union => {
                        // pointer to anonymous struct to pointer to union
                        if (inst_ty.isSinglePointer() and
                            inst_ty.childType().isAnonStruct() and
                            !dest_info.mutable)
                        {
                            return sema.coerceAnonStructToUnionPtrs(block, dest_ty, dest_ty_src, inst, inst_src);
                        }
                    },
                    .Struct => {
                        // pointer to anonymous struct to pointer to struct
                        if (inst_ty.isSinglePointer() and
                            inst_ty.childType().isAnonStruct() and
                            !dest_info.mutable)
                        {
                            return sema.coerceAnonStructToStructPtrs(block, dest_ty, dest_ty_src, inst, inst_src);
                        }
                    },
                    .Array => {
                        // pointer to tuple to pointer to array
                        if (inst_ty.isSinglePointer() and
                            inst_ty.childType().isTuple() and
                            !dest_info.mutable)
                        {
                            return sema.coerceTupleToArrayPtrs(block, dest_ty, dest_ty_src, inst, inst_src);
                        }
                    },
                    else => {},
                },
                .Slice => {
                    // pointer to tuple to slice
                    if (inst_ty.isSinglePointer() and
                        inst_ty.childType().isTuple() and
                        !dest_info.mutable and dest_info.size == .Slice)
                    {
                        return sema.coerceTupleToSlicePtrs(block, dest_ty, dest_ty_src, inst, inst_src);
                    }
                },
                .Many => p: {
                    if (!inst_ty.isSlice()) break :p;
                    const inst_info = inst_ty.ptrInfo().data;

                    switch (try sema.coerceInMemoryAllowed(
                        block,
                        dest_info.pointee_type,
                        inst_info.pointee_type,
                        dest_info.mutable,
                        target,
                        dest_ty_src,
                        inst_src,
                    )) {
                        .ok => {},
                        .no_match => break :p,
                    }

                    if (dest_info.sentinel == null or inst_info.sentinel == null or
                        !dest_info.sentinel.?.eql(inst_info.sentinel.?, dest_info.pointee_type, target))
                        break :p;

                    const slice_ptr = try sema.analyzeSlicePtr(block, inst_src, inst, inst_ty);
                    return sema.coerceCompatiblePtrs(block, dest_ty, slice_ptr, inst_src);
                },
            }
        },
        .Int, .ComptimeInt => switch (inst_ty.zigTypeTag()) {
            .Float, .ComptimeFloat => float: {
                const val = (try sema.resolveDefinedValue(block, inst_src, inst)) orelse break :float;

                if (val.floatHasFraction()) {
                    return sema.fail(block, inst_src, "fractional component prevents float value {} from coercion to type '{}'", .{ val.fmtValue(inst_ty, target), dest_ty.fmt(target) });
                }
                const result_val = val.floatToInt(sema.arena, inst_ty, dest_ty, target) catch |err| switch (err) {
                    error.FloatCannotFit => {
                        return sema.fail(block, inst_src, "integer value {d} cannot be stored in type '{}'", .{ std.math.floor(val.toFloat(f64)), dest_ty.fmt(target) });
                    },
                    else => |e| return e,
                };
                return try sema.addConstant(dest_ty, result_val);
            },
            .Int, .ComptimeInt => {
                if (try sema.resolveDefinedValue(block, inst_src, inst)) |val| {
                    // comptime known integer to other number
                    if (!val.intFitsInType(dest_ty, target)) {
                        return sema.fail(block, inst_src, "type {} cannot represent integer value {}", .{ dest_ty.fmt(target), val.fmtValue(inst_ty, target) });
                    }
                    return try sema.addConstant(dest_ty, val);
                }

                // integer widening
                const dst_info = dest_ty.intInfo(target);
                const src_info = inst_ty.intInfo(target);
                if ((src_info.signedness == dst_info.signedness and dst_info.bits >= src_info.bits) or
                    // small enough unsigned ints can get casted to large enough signed ints
                    (dst_info.signedness == .signed and dst_info.bits > src_info.bits))
                {
                    try sema.requireRuntimeBlock(block, inst_src);
                    return block.addTyOp(.intcast, dest_ty, inst);
                }
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .Float, .ComptimeFloat => switch (inst_ty.zigTypeTag()) {
            .ComptimeFloat => {
                const val = try sema.resolveConstValue(block, inst_src, inst);
                const result_val = try val.floatCast(sema.arena, dest_ty, target);
                return try sema.addConstant(dest_ty, result_val);
            },
            .Float => {
                if (try sema.resolveDefinedValue(block, inst_src, inst)) |val| {
                    const result_val = try val.floatCast(sema.arena, dest_ty, target);
                    if (!val.eql(result_val, dest_ty, target)) {
                        return sema.fail(
                            block,
                            inst_src,
                            "type {} cannot represent float value {}",
                            .{ dest_ty.fmt(target), val.fmtValue(inst_ty, target) },
                        );
                    }
                    return try sema.addConstant(dest_ty, result_val);
                }

                // float widening
                const src_bits = inst_ty.floatBits(target);
                const dst_bits = dest_ty.floatBits(target);
                if (dst_bits >= src_bits) {
                    try sema.requireRuntimeBlock(block, inst_src);
                    return block.addTyOp(.fpext, dest_ty, inst);
                }
            },
            .Int, .ComptimeInt => int: {
                const val = (try sema.resolveDefinedValue(block, inst_src, inst)) orelse break :int;
                const result_val = try val.intToFloat(sema.arena, inst_ty, dest_ty, target);
                // TODO implement this compile error
                //const int_again_val = try result_val.floatToInt(sema.arena, inst_ty);
                //if (!int_again_val.eql(val, inst_ty, target)) {
                //    return sema.fail(
                //        block,
                //        inst_src,
                //        "type {} cannot represent integer value {}",
                //        .{ dest_ty.fmt(target), val },
                //    );
                //}
                return try sema.addConstant(dest_ty, result_val);
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .Enum => switch (inst_ty.zigTypeTag()) {
            .EnumLiteral => {
                // enum literal to enum
                const val = try sema.resolveConstValue(block, inst_src, inst);
                const bytes = val.castTag(.enum_literal).?.data;
                const field_index = dest_ty.enumFieldIndex(bytes) orelse {
                    const msg = msg: {
                        const msg = try sema.errMsg(
                            block,
                            inst_src,
                            "enum '{}' has no field named '{s}'",
                            .{ dest_ty.fmt(target), bytes },
                        );
                        errdefer msg.destroy(sema.gpa);
                        try sema.mod.errNoteNonLazy(
                            dest_ty.declSrcLoc(),
                            msg,
                            "enum declared here",
                            .{},
                        );
                        break :msg msg;
                    };
                    return sema.failWithOwnedErrorMsg(block, msg);
                };
                return sema.addConstant(
                    dest_ty,
                    try Value.Tag.enum_field_index.create(arena, @intCast(u32, field_index)),
                );
            },
            .Union => blk: {
                // union to its own tag type
                const union_tag_ty = inst_ty.unionTagType() orelse break :blk;
                if (union_tag_ty.eql(dest_ty, target)) {
                    return sema.unionToTag(block, dest_ty, inst, inst_src);
                }
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .ErrorUnion => switch (inst_ty.zigTypeTag()) {
            .ErrorUnion => {
                if (maybe_inst_val) |inst_val| {
                    switch (inst_val.tag()) {
                        .undef => return sema.addConstUndef(dest_ty),
                        .eu_payload => {
                            const payload = try sema.addConstant(
                                inst_ty.errorUnionPayload(),
                                inst_val.castTag(.eu_payload).?.data,
                            );
                            return sema.wrapErrorUnionPayload(block, dest_ty, payload, inst_src);
                        },
                        else => {
                            const error_set = try sema.addConstant(
                                inst_ty.errorUnionSet(),
                                inst_val,
                            );
                            return sema.wrapErrorUnionSet(block, dest_ty, error_set, inst_src);
                        },
                    }
                }
            },
            .ErrorSet => {
                // E to E!T
                return sema.wrapErrorUnionSet(block, dest_ty, inst, inst_src);
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {
                // T to E!T
                return sema.wrapErrorUnionPayload(block, dest_ty, inst, inst_src);
            },
        },
        .Union => switch (inst_ty.zigTypeTag()) {
            .Enum, .EnumLiteral => return sema.coerceEnumToUnion(block, dest_ty, dest_ty_src, inst, inst_src),
            .Struct => {
                if (inst_ty.isAnonStruct()) {
                    return sema.coerceAnonStructToUnion(block, dest_ty, dest_ty_src, inst, inst_src);
                }
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .Array => switch (inst_ty.zigTypeTag()) {
            .Vector => return sema.coerceArrayLike(block, dest_ty, dest_ty_src, inst, inst_src),
            .Struct => {
                if (inst == .empty_struct) {
                    return arrayInitEmpty(sema, dest_ty);
                }
                if (inst_ty.isTuple()) {
                    return sema.coerceTupleToArray(block, dest_ty, dest_ty_src, inst, inst_src);
                }
            },
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .Vector => switch (inst_ty.zigTypeTag()) {
            .Array, .Vector => return sema.coerceArrayLike(block, dest_ty, dest_ty_src, inst, inst_src),
            .Undefined => {
                return sema.addConstUndef(dest_ty);
            },
            else => {},
        },
        .Struct => {
            if (inst == .empty_struct) {
                return structInitEmpty(sema, block, dest_ty, dest_ty_src, inst_src);
            }
            if (inst_ty.isTupleOrAnonStruct()) {
                return sema.coerceTupleToStruct(block, dest_ty, dest_ty_src, inst, inst_src);
            }
        },
        else => {},
    }

    // undefined to anything. We do this after the big switch above so that
    // special logic has a chance to run first, such as `*[N]T` to `[]T` which
    // should initialize the length field of the slice.
    if (is_undef) {
        return sema.addConstUndef(dest_ty);
    }

    return sema.fail(block, inst_src, "expected {}, found {}", .{ dest_ty.fmt(target), inst_ty.fmt(target) });
}

const InMemoryCoercionResult = enum {
    ok,
    no_match,
};

/// If pointers have the same representation in runtime memory, a bitcast AIR instruction
/// may be used for the coercion.
/// * `const` attribute can be gained
/// * `volatile` attribute can be gained
/// * `allowzero` attribute can be gained (whether from explicit attribute, C pointer, or optional pointer) but only if !dest_is_mut
/// * alignment can be decreased
/// * bit offset attributes must match exactly
/// * `*`/`[*]` must match exactly, but `[*c]` matches either one
/// * sentinel-terminated pointers can coerce into `[*]`
/// TODO improve this function to report recursive compile errors like it does in stage1.
/// look at the function types_match_const_cast_only
fn coerceInMemoryAllowed(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    src_ty: Type,
    dest_is_mut: bool,
    target: std.Target,
    dest_src: LazySrcLoc,
    src_src: LazySrcLoc,
) CompileError!InMemoryCoercionResult {
    if (dest_ty.eql(src_ty, target))
        return .ok;

    // Pointers / Pointer-like Optionals
    var dest_buf: Type.Payload.ElemType = undefined;
    var src_buf: Type.Payload.ElemType = undefined;
    const maybe_dest_ptr_ty = try sema.typePtrOrOptionalPtrTy(block, dest_ty, &dest_buf, dest_src);
    const maybe_src_ptr_ty = try sema.typePtrOrOptionalPtrTy(block, src_ty, &src_buf, src_src);
    if (maybe_dest_ptr_ty) |dest_ptr_ty| {
        if (maybe_src_ptr_ty) |src_ptr_ty| {
            return try sema.coerceInMemoryAllowedPtrs(block, dest_ty, src_ty, dest_ptr_ty, src_ptr_ty, dest_is_mut, target, dest_src, src_src);
        }
    }

    // Slices
    if (dest_ty.isSlice() and src_ty.isSlice()) {
        return try sema.coerceInMemoryAllowedPtrs(block, dest_ty, src_ty, dest_ty, src_ty, dest_is_mut, target, dest_src, src_src);
    }

    const dest_tag = dest_ty.zigTypeTag();
    const src_tag = src_ty.zigTypeTag();

    // Functions
    if (dest_tag == .Fn and src_tag == .Fn) {
        return try sema.coerceInMemoryAllowedFns(block, dest_ty, src_ty, target, dest_src, src_src);
    }

    // Error Unions
    if (dest_tag == .ErrorUnion and src_tag == .ErrorUnion) {
        const child = try sema.coerceInMemoryAllowed(block, dest_ty.errorUnionPayload(), src_ty.errorUnionPayload(), dest_is_mut, target, dest_src, src_src);
        if (child == .no_match) {
            return child;
        }
        return try sema.coerceInMemoryAllowed(block, dest_ty.errorUnionSet(), src_ty.errorUnionSet(), dest_is_mut, target, dest_src, src_src);
    }

    // Error Sets
    if (dest_tag == .ErrorSet and src_tag == .ErrorSet) {
        return try sema.coerceInMemoryAllowedErrorSets(block, dest_ty, src_ty, dest_src, src_src);
    }

    // Arrays
    if (dest_tag == .Array and src_tag == .Array) arrays: {
        const dest_info = dest_ty.arrayInfo();
        const src_info = src_ty.arrayInfo();
        if (dest_info.len != src_info.len) break :arrays;

        const child = try sema.coerceInMemoryAllowed(block, dest_info.elem_type, src_info.elem_type, dest_is_mut, target, dest_src, src_src);
        if (child == .no_match) {
            return child;
        }
        const ok_sent = dest_info.sentinel == null or
            (src_info.sentinel != null and
            dest_info.sentinel.?.eql(src_info.sentinel.?, dest_info.elem_type, target));
        if (!ok_sent) {
            return .no_match;
        }
        return .ok;
    }

    // Vectors
    if (dest_tag == .Vector and src_tag == .Vector) vectors: {
        const dest_len = dest_ty.vectorLen();
        const src_len = src_ty.vectorLen();
        if (dest_len != src_len) break :vectors;

        const dest_elem_ty = dest_ty.scalarType();
        const src_elem_ty = src_ty.scalarType();
        const child = try sema.coerceInMemoryAllowed(block, dest_elem_ty, src_elem_ty, dest_is_mut, target, dest_src, src_src);
        if (child == .no_match) break :vectors;

        return .ok;
    }

    // Optionals
    if (dest_tag == .Optional and src_tag == .Optional) optionals: {
        if ((maybe_dest_ptr_ty != null) != (maybe_src_ptr_ty != null)) {
            // TODO "optional type child '{}' cannot cast into optional type '{}'"
            return .no_match;
        }
        const dest_child_type = dest_ty.optionalChild(&dest_buf);
        const src_child_type = src_ty.optionalChild(&src_buf);

        const child = try sema.coerceInMemoryAllowed(block, dest_child_type, src_child_type, dest_is_mut, target, dest_src, src_src);
        if (child == .no_match) {
            // TODO "optional type child '{}' cannot cast into optional type child '{}'"
            break :optionals;
        }

        return .ok;
    }

    return .no_match;
}

fn coerceInMemoryAllowedErrorSets(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    src_ty: Type,
    dest_src: LazySrcLoc,
    src_src: LazySrcLoc,
) !InMemoryCoercionResult {
    // Coercion to `anyerror`. Note that this check can return false negatives
    // in case the error sets did not get resolved.
    if (dest_ty.isAnyError()) {
        return .ok;
    }

    if (dest_ty.castTag(.error_set_inferred)) |dst_payload| {
        const dst_ies = dst_payload.data;
        // We will make an effort to return `ok` without resolving either error set, to
        // avoid unnecessary "unable to resolve error set" dependency loop errors.
        switch (src_ty.tag()) {
            .error_set_inferred => {
                // If both are inferred error sets of functions, and
                // the dest includes the source function, the coercion is OK.
                // This check is important because it works without forcing a full resolution
                // of inferred error sets.
                const src_ies = src_ty.castTag(.error_set_inferred).?.data;

                if (dst_ies.inferred_error_sets.contains(src_ies)) {
                    return .ok;
                }
            },
            .error_set_single => {
                const name = src_ty.castTag(.error_set_single).?.data;
                if (dst_ies.errors.contains(name)) return .ok;
            },
            .error_set_merged => {
                const names = src_ty.castTag(.error_set_merged).?.data.keys();
                for (names) |name| {
                    if (!dst_ies.errors.contains(name)) break;
                } else return .ok;
            },
            .error_set => {
                const names = src_ty.castTag(.error_set).?.data.names.keys();
                for (names) |name| {
                    if (!dst_ies.errors.contains(name)) break;
                } else return .ok;
            },
            .anyerror => {},
            else => unreachable,
        }

        if (dst_ies.func == sema.owner_func) {
            // We are trying to coerce an error set to the current function's
            // inferred error set.
            try dst_ies.addErrorSet(sema.gpa, src_ty);
            return .ok;
        }

        try sema.resolveInferredErrorSet(block, dest_src, dst_payload.data);
        // isAnyError might have changed from a false negative to a true positive after resolution.
        if (dest_ty.isAnyError()) {
            return .ok;
        }
    }

    switch (src_ty.tag()) {
        .error_set_inferred => {
            const src_data = src_ty.castTag(.error_set_inferred).?.data;

            try sema.resolveInferredErrorSet(block, src_src, src_data);
            // src anyerror status might have changed after the resolution.
            if (src_ty.isAnyError()) {
                // dest_ty.isAnyError() == true is already checked for at this point.
                return .no_match;
            }

            for (src_data.errors.keys()) |key| {
                if (!dest_ty.errorSetHasField(key)) {
                    return .no_match;
                }
            }

            return .ok;
        },
        .error_set_single => {
            const name = src_ty.castTag(.error_set_single).?.data;
            if (dest_ty.errorSetHasField(name)) {
                return .ok;
            }
        },
        .error_set_merged => {
            const names = src_ty.castTag(.error_set_merged).?.data.keys();
            for (names) |name| {
                if (!dest_ty.errorSetHasField(name)) {
                    return .no_match;
                }
            }

            return .ok;
        },
        .error_set => {
            const names = src_ty.castTag(.error_set).?.data.names.keys();
            for (names) |name| {
                if (!dest_ty.errorSetHasField(name)) {
                    return .no_match;
                }
            }

            return .ok;
        },
        .anyerror => switch (dest_ty.tag()) {
            .error_set_inferred => return .no_match, // Caught by dest.isAnyError() above.
            .error_set_single, .error_set_merged, .error_set => {},
            .anyerror => unreachable, // Filtered out above.
            else => unreachable,
        },
        else => unreachable,
    }

    return .no_match;
}

fn coerceInMemoryAllowedFns(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    src_ty: Type,
    target: std.Target,
    dest_src: LazySrcLoc,
    src_src: LazySrcLoc,
) !InMemoryCoercionResult {
    const dest_info = dest_ty.fnInfo();
    const src_info = src_ty.fnInfo();

    if (dest_info.is_var_args != src_info.is_var_args) {
        return .no_match;
    }

    if (dest_info.is_generic != src_info.is_generic) {
        return .no_match;
    }

    if (!src_info.return_type.isNoReturn()) {
        const rt = try sema.coerceInMemoryAllowed(block, dest_info.return_type, src_info.return_type, false, target, dest_src, src_src);
        if (rt == .no_match) {
            return rt;
        }
    }

    if (dest_info.param_types.len != src_info.param_types.len) {
        return .no_match;
    }

    for (dest_info.param_types) |dest_param_ty, i| {
        const src_param_ty = src_info.param_types[i];

        if (dest_info.comptime_params[i] != src_info.comptime_params[i]) {
            return .no_match;
        }

        // TODO: noalias

        // Note: Cast direction is reversed here.
        const param = try sema.coerceInMemoryAllowed(block, src_param_ty, dest_param_ty, false, target, dest_src, src_src);
        if (param == .no_match) {
            return param;
        }
    }

    if (dest_info.cc != src_info.cc) {
        return .no_match;
    }

    return .ok;
}

fn coerceInMemoryAllowedPtrs(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    src_ty: Type,
    dest_ptr_ty: Type,
    src_ptr_ty: Type,
    dest_is_mut: bool,
    target: std.Target,
    dest_src: LazySrcLoc,
    src_src: LazySrcLoc,
) !InMemoryCoercionResult {
    const dest_info = dest_ptr_ty.ptrInfo().data;
    const src_info = src_ptr_ty.ptrInfo().data;

    const child = try sema.coerceInMemoryAllowed(block, dest_info.pointee_type, src_info.pointee_type, dest_info.mutable, target, dest_src, src_src);
    if (child == .no_match) {
        return child;
    }

    if (dest_info.@"addrspace" != src_info.@"addrspace") {
        return .no_match;
    }

    const ok_sent = dest_info.sentinel == null or src_info.size == .C or
        (src_info.sentinel != null and
        dest_info.sentinel.?.eql(src_info.sentinel.?, dest_info.pointee_type, target));
    if (!ok_sent) {
        return .no_match;
    }

    const ok_ptr_size = src_info.size == dest_info.size or
        src_info.size == .C or dest_info.size == .C;
    if (!ok_ptr_size) {
        return .no_match;
    }

    const ok_cv_qualifiers =
        (src_info.mutable or !dest_info.mutable) and
        (!src_info.@"volatile" or dest_info.@"volatile");

    if (!ok_cv_qualifiers) {
        return .no_match;
    }

    const dest_allow_zero = dest_ty.ptrAllowsZero();
    const src_allow_zero = src_ty.ptrAllowsZero();

    const ok_allows_zero = (dest_allow_zero and
        (src_allow_zero or !dest_is_mut)) or
        (!dest_allow_zero and !src_allow_zero);
    if (!ok_allows_zero) {
        return .no_match;
    }

    if (src_info.host_size != dest_info.host_size or
        src_info.bit_offset != dest_info.bit_offset)
    {
        return .no_match;
    }

    // If both pointers have alignment 0, it means they both want ABI alignment.
    // In this case, if they share the same child type, no need to resolve
    // pointee type alignment. Otherwise both pointee types must have their alignment
    // resolved and we compare the alignment numerically.
    alignment: {
        if (src_info.@"align" == 0 and dest_info.@"align" == 0 and
            dest_info.pointee_type.eql(src_info.pointee_type, target))
        {
            break :alignment;
        }

        const src_align = if (src_info.@"align" != 0)
            src_info.@"align"
        else
            src_info.pointee_type.abiAlignment(target);

        const dest_align = if (dest_info.@"align" != 0)
            dest_info.@"align"
        else
            dest_info.pointee_type.abiAlignment(target);

        if (dest_align > src_align) {
            return .no_match;
        }

        break :alignment;
    }

    return .ok;
}

fn coerceVarArgParam(
    sema: *Sema,
    block: *Block,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    switch (inst_ty.zigTypeTag()) {
        .ComptimeInt, .ComptimeFloat => return sema.fail(block, inst_src, "integer and float literals in var args function must be casted", .{}),
        else => {},
    }
    // TODO implement more of this function.
    return inst;
}

// TODO migrate callsites to use storePtr2 instead.
fn storePtr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr: Air.Inst.Ref,
    uncasted_operand: Air.Inst.Ref,
) CompileError!void {
    return sema.storePtr2(block, src, ptr, src, uncasted_operand, src, .store);
}

fn storePtr2(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr: Air.Inst.Ref,
    ptr_src: LazySrcLoc,
    uncasted_operand: Air.Inst.Ref,
    operand_src: LazySrcLoc,
    air_tag: Air.Inst.Tag,
) CompileError!void {
    const ptr_ty = sema.typeOf(ptr);
    if (ptr_ty.isConstPtr())
        return sema.fail(block, ptr_src, "cannot assign to constant", .{});

    const elem_ty = ptr_ty.childType();

    // To generate better code for tuples, we detect a tuple operand here, and
    // analyze field loads and stores directly. This avoids an extra allocation + memcpy
    // which would occur if we used `coerce`.
    // However, we avoid this mechanism if the destination element type is a tuple,
    // because the regular store will be better for this case.
    // If the destination type is a struct we don't want this mechanism to trigger, because
    // this code does not handle tuple-to-struct coercion which requires dealing with missing
    // fields.
    const operand_ty = sema.typeOf(uncasted_operand);
    if (operand_ty.isTuple() and elem_ty.zigTypeTag() == .Array) {
        const tuple = operand_ty.tupleFields();
        for (tuple.types) |_, i_usize| {
            const i = @intCast(u32, i_usize);
            const elem_src = operand_src; // TODO better source location
            const elem = try tupleField(sema, block, operand_src, uncasted_operand, elem_src, i);
            const elem_index = try sema.addIntUnsigned(Type.usize, i);
            const elem_ptr = try sema.elemPtr(block, ptr_src, ptr, elem_index, elem_src);
            try sema.storePtr2(block, src, elem_ptr, elem_src, elem, elem_src, .store);
        }
        return;
    }

    // TODO do the same thing for anon structs as for tuples above.
    // However, beware of the need to handle missing/extra fields.

    // Detect if we are storing an array operand to a bitcasted vector pointer.
    // If so, we instead reach through the bitcasted pointer to the vector pointer,
    // bitcast the array operand to a vector, and then lower this as a store of
    // a vector value to a vector pointer. This generally results in better code,
    // as well as working around an LLVM bug:
    // https://github.com/ziglang/zig/issues/11154
    if (sema.obtainBitCastedVectorPtr(ptr)) |vector_ptr| {
        const vector_ty = sema.typeOf(vector_ptr).childType();
        const vector = try sema.coerce(block, vector_ty, uncasted_operand, operand_src);
        try sema.storePtr2(block, src, vector_ptr, ptr_src, vector, operand_src, .store);
        return;
    }

    const operand = try sema.coerce(block, elem_ty, uncasted_operand, operand_src);
    const maybe_operand_val = try sema.resolveMaybeUndefVal(block, operand_src, operand);

    const runtime_src = if (try sema.resolveDefinedValue(block, ptr_src, ptr)) |ptr_val| rs: {
        const operand_val = maybe_operand_val orelse {
            try sema.checkPtrIsNotComptimeMutable(block, ptr_val, ptr_src, operand_src);
            break :rs operand_src;
        };
        if (ptr_val.isComptimeMutablePtr()) {
            try sema.storePtrVal(block, src, ptr_val, operand_val, elem_ty);
            return;
        } else break :rs ptr_src;
    } else ptr_src;

    // We do this after the possible comptime store above, for the case of field_ptr stores
    // to unions because we want the comptime tag to be set, even if the field type is void.
    if ((try sema.typeHasOnePossibleValue(block, src, elem_ty)) != null)
        return;

    // TODO handle if the element type requires comptime

    try sema.requireRuntimeBlock(block, runtime_src);
    try sema.queueFullTypeResolution(elem_ty);
    _ = try block.addBinOp(air_tag, ptr, operand);
}

/// Traverse an arbitrary number of bitcasted pointers and return the underyling vector
/// pointer. Only if the final element type matches the vector element type, and the
/// lengths match.
fn obtainBitCastedVectorPtr(sema: *Sema, ptr: Air.Inst.Ref) ?Air.Inst.Ref {
    const array_ty = sema.typeOf(ptr).childType();
    if (array_ty.zigTypeTag() != .Array) return null;
    var ptr_inst = Air.refToIndex(ptr) orelse return null;
    const air_datas = sema.air_instructions.items(.data);
    const air_tags = sema.air_instructions.items(.tag);
    const prev_ptr = while (air_tags[ptr_inst] == .bitcast) {
        const prev_ptr = air_datas[ptr_inst].ty_op.operand;
        const prev_ptr_ty = sema.typeOf(prev_ptr);
        const prev_ptr_child_ty = switch (prev_ptr_ty.tag()) {
            .single_mut_pointer => prev_ptr_ty.castTag(.single_mut_pointer).?.data,
            .pointer => prev_ptr_ty.castTag(.pointer).?.data.pointee_type,
            else => return null,
        };
        if (prev_ptr_child_ty.zigTypeTag() == .Vector) break prev_ptr;
        ptr_inst = Air.refToIndex(prev_ptr) orelse return null;
    } else return null;

    // We have a pointer-to-array and a pointer-to-vector. If the elements and
    // lengths match, return the result.
    const vector_ty = sema.typeOf(prev_ptr).childType();
    const target = sema.mod.getTarget();
    if (array_ty.childType().eql(vector_ty.childType(), target) and
        array_ty.arrayLen() == vector_ty.vectorLen())
    {
        return prev_ptr;
    } else {
        return null;
    }
}

/// Call when you have Value objects rather than Air instructions, and you want to
/// assert the store must be done at comptime.
fn storePtrVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr_val: Value,
    operand_val: Value,
    operand_ty: Type,
) !void {
    var mut_kit = try beginComptimePtrMutation(sema, block, src, ptr_val);
    try sema.checkComptimeVarStore(block, src, mut_kit.decl_ref_mut);

    const bitcasted_val = try sema.bitCastVal(block, src, operand_val, operand_ty, mut_kit.ty, 0);

    const arena = mut_kit.beginArena(sema.gpa);
    defer mut_kit.finishArena();

    mut_kit.val.* = try bitcasted_val.copy(arena);
}

const ComptimePtrMutationKit = struct {
    decl_ref_mut: Value.Payload.DeclRefMut.Data,
    val: *Value,
    ty: Type,
    decl_arena: std.heap.ArenaAllocator = undefined,

    fn beginArena(self: *ComptimePtrMutationKit, gpa: Allocator) Allocator {
        self.decl_arena = self.decl_ref_mut.decl.value_arena.?.promote(gpa);
        return self.decl_arena.allocator();
    }

    fn finishArena(self: *ComptimePtrMutationKit) void {
        self.decl_ref_mut.decl.value_arena.?.* = self.decl_arena.state;
        self.decl_arena = undefined;
    }
};

fn beginComptimePtrMutation(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr_val: Value,
) CompileError!ComptimePtrMutationKit {

    // TODO: Update this to behave like `beginComptimePtrLoad` and properly check/use
    // `container_ty` and `array_ty`, instead of trusting that the parent decl type
    // matches the type used to derive the elem_ptr/field_ptr/etc.
    //
    // This is needed because the types will not match if the pointer we're mutating
    // through is reinterpreting comptime memory.

    switch (ptr_val.tag()) {
        .decl_ref_mut => {
            const decl_ref_mut = ptr_val.castTag(.decl_ref_mut).?.data;
            return ComptimePtrMutationKit{
                .decl_ref_mut = decl_ref_mut,
                .val = &decl_ref_mut.decl.val,
                .ty = decl_ref_mut.decl.ty,
            };
        },
        .elem_ptr => {
            const elem_ptr = ptr_val.castTag(.elem_ptr).?.data;
            var parent = try beginComptimePtrMutation(sema, block, src, elem_ptr.array_ptr);
            switch (parent.ty.zigTypeTag()) {
                .Array, .Vector => {
                    const check_len = parent.ty.arrayLenIncludingSentinel();
                    if (elem_ptr.index >= check_len) {
                        // TODO have the parent include the decl so we can say "declared here"
                        return sema.fail(block, src, "comptime store of index {d} out of bounds of array length {d}", .{
                            elem_ptr.index, check_len,
                        });
                    }
                    const elem_ty = parent.ty.childType();
                    switch (parent.val.tag()) {
                        .undef => {
                            // An array has been initialized to undefined at comptime and now we
                            // are for the first time setting an element. We must change the representation
                            // of the array from `undef` to `array`.
                            const arena = parent.beginArena(sema.gpa);
                            defer parent.finishArena();

                            const array_len_including_sentinel =
                                try sema.usizeCast(block, src, parent.ty.arrayLenIncludingSentinel());
                            const elems = try arena.alloc(Value, array_len_including_sentinel);
                            mem.set(Value, elems, Value.undef);

                            parent.val.* = try Value.Tag.aggregate.create(arena, elems);

                            return ComptimePtrMutationKit{
                                .decl_ref_mut = parent.decl_ref_mut,
                                .val = &elems[elem_ptr.index],
                                .ty = elem_ty,
                            };
                        },
                        .bytes => {
                            // An array is memory-optimized to store a slice of bytes, but we are about
                            // to modify an individual field and the representation has to change.
                            // If we wanted to avoid this, there would need to be special detection
                            // elsewhere to identify when writing a value to an array element that is stored
                            // using the `bytes` tag, and handle it without making a call to this function.
                            const arena = parent.beginArena(sema.gpa);
                            defer parent.finishArena();

                            const bytes = parent.val.castTag(.bytes).?.data;
                            const dest_len = parent.ty.arrayLenIncludingSentinel();
                            // bytes.len may be one greater than dest_len because of the case when
                            // assigning `[N:S]T` to `[N]T`. This is allowed; the sentinel is omitted.
                            assert(bytes.len >= dest_len);
                            const elems = try arena.alloc(Value, @intCast(usize, dest_len));
                            for (elems) |*elem, i| {
                                elem.* = try Value.Tag.int_u64.create(arena, bytes[i]);
                            }

                            parent.val.* = try Value.Tag.aggregate.create(arena, elems);

                            return ComptimePtrMutationKit{
                                .decl_ref_mut = parent.decl_ref_mut,
                                .val = &elems[elem_ptr.index],
                                .ty = elem_ty,
                            };
                        },
                        .repeated => {
                            // An array is memory-optimized to store only a single element value, and
                            // that value is understood to be the same for the entire length of the array.
                            // However, now we want to modify an individual field and so the
                            // representation has to change.  If we wanted to avoid this, there would
                            // need to be special detection elsewhere to identify when writing a value to an
                            // array element that is stored using the `repeated` tag, and handle it
                            // without making a call to this function.
                            const arena = parent.beginArena(sema.gpa);
                            defer parent.finishArena();

                            const repeated_val = try parent.val.castTag(.repeated).?.data.copy(arena);
                            const array_len_including_sentinel =
                                try sema.usizeCast(block, src, parent.ty.arrayLenIncludingSentinel());
                            const elems = try arena.alloc(Value, array_len_including_sentinel);
                            mem.set(Value, elems, repeated_val);

                            parent.val.* = try Value.Tag.aggregate.create(arena, elems);

                            return ComptimePtrMutationKit{
                                .decl_ref_mut = parent.decl_ref_mut,
                                .val = &elems[elem_ptr.index],
                                .ty = elem_ty,
                            };
                        },

                        .aggregate => return ComptimePtrMutationKit{
                            .decl_ref_mut = parent.decl_ref_mut,
                            .val = &parent.val.castTag(.aggregate).?.data[elem_ptr.index],
                            .ty = elem_ty,
                        },

                        else => unreachable,
                    }
                },
                else => {
                    if (elem_ptr.index != 0) {
                        // TODO include a "declared here" note for the decl
                        return sema.fail(block, src, "out of bounds comptime store of index {d}", .{
                            elem_ptr.index,
                        });
                    }
                    return ComptimePtrMutationKit{
                        .decl_ref_mut = parent.decl_ref_mut,
                        .val = parent.val,
                        .ty = parent.ty,
                    };
                },
            }
        },
        .field_ptr => {
            const field_ptr = ptr_val.castTag(.field_ptr).?.data;
            var parent = try beginComptimePtrMutation(sema, block, src, field_ptr.container_ptr);
            const field_index = @intCast(u32, field_ptr.field_index);
            const field_ty = parent.ty.structFieldType(field_index);
            switch (parent.val.tag()) {
                .undef => {
                    // A struct or union has been initialized to undefined at comptime and now we
                    // are for the first time setting a field. We must change the representation
                    // of the struct/union from `undef` to `struct`/`union`.
                    const arena = parent.beginArena(sema.gpa);
                    defer parent.finishArena();

                    switch (parent.ty.zigTypeTag()) {
                        .Struct => {
                            const fields = try arena.alloc(Value, parent.ty.structFieldCount());
                            mem.set(Value, fields, Value.undef);

                            parent.val.* = try Value.Tag.aggregate.create(arena, fields);

                            return ComptimePtrMutationKit{
                                .decl_ref_mut = parent.decl_ref_mut,
                                .val = &fields[field_index],
                                .ty = field_ty,
                            };
                        },
                        .Union => {
                            const payload = try arena.create(Value.Payload.Union);
                            payload.* = .{ .data = .{
                                .tag = try Value.Tag.enum_field_index.create(arena, field_index),
                                .val = Value.undef,
                            } };

                            parent.val.* = Value.initPayload(&payload.base);

                            return ComptimePtrMutationKit{
                                .decl_ref_mut = parent.decl_ref_mut,
                                .val = &payload.data.val,
                                .ty = field_ty,
                            };
                        },
                        else => unreachable,
                    }
                },
                .aggregate => return ComptimePtrMutationKit{
                    .decl_ref_mut = parent.decl_ref_mut,
                    .val = &parent.val.castTag(.aggregate).?.data[field_index],
                    .ty = field_ty,
                },
                .@"union" => {
                    // We need to set the active field of the union.
                    const arena = parent.beginArena(sema.gpa);
                    defer parent.finishArena();

                    const payload = &parent.val.castTag(.@"union").?.data;
                    payload.tag = try Value.Tag.enum_field_index.create(arena, field_index);

                    return ComptimePtrMutationKit{
                        .decl_ref_mut = parent.decl_ref_mut,
                        .val = &payload.val,
                        .ty = field_ty,
                    };
                },

                else => unreachable,
            }
        },
        .eu_payload_ptr => {
            const eu_ptr = ptr_val.castTag(.eu_payload_ptr).?.data;
            var parent = try beginComptimePtrMutation(sema, block, src, eu_ptr.container_ptr);
            const payload_ty = parent.ty.errorUnionPayload();
            switch (parent.val.tag()) {
                else => {
                    // An error union has been initialized to undefined at comptime and now we
                    // are for the first time setting the payload. We must change the
                    // representation of the error union from `undef` to `opt_payload`.
                    const arena = parent.beginArena(sema.gpa);
                    defer parent.finishArena();

                    const payload = try arena.create(Value.Payload.SubValue);
                    payload.* = .{
                        .base = .{ .tag = .eu_payload },
                        .data = Value.undef,
                    };

                    parent.val.* = Value.initPayload(&payload.base);

                    return ComptimePtrMutationKit{
                        .decl_ref_mut = parent.decl_ref_mut,
                        .val = &payload.data,
                        .ty = payload_ty,
                    };
                },
                .eu_payload => return ComptimePtrMutationKit{
                    .decl_ref_mut = parent.decl_ref_mut,
                    .val = &parent.val.castTag(.eu_payload).?.data,
                    .ty = payload_ty,
                },
            }
        },
        .opt_payload_ptr => {
            const opt_ptr = ptr_val.castTag(.opt_payload_ptr).?.data;
            var parent = try beginComptimePtrMutation(sema, block, src, opt_ptr.container_ptr);
            const payload_ty = try parent.ty.optionalChildAlloc(sema.arena);
            switch (parent.val.tag()) {
                .undef, .null_value => {
                    // An optional has been initialized to undefined at comptime and now we
                    // are for the first time setting the payload. We must change the
                    // representation of the optional from `undef` to `opt_payload`.
                    const arena = parent.beginArena(sema.gpa);
                    defer parent.finishArena();

                    const payload = try arena.create(Value.Payload.SubValue);
                    payload.* = .{
                        .base = .{ .tag = .opt_payload },
                        .data = Value.undef,
                    };

                    parent.val.* = Value.initPayload(&payload.base);

                    return ComptimePtrMutationKit{
                        .decl_ref_mut = parent.decl_ref_mut,
                        .val = &payload.data,
                        .ty = payload_ty,
                    };
                },
                .opt_payload => return ComptimePtrMutationKit{
                    .decl_ref_mut = parent.decl_ref_mut,
                    .val = &parent.val.castTag(.opt_payload).?.data,
                    .ty = payload_ty,
                },

                else => unreachable,
            }
        },
        .decl_ref => unreachable, // isComptimeMutablePtr() has been checked already
        else => unreachable,
    }
}

const TypedValueAndOffset = struct {
    tv: TypedValue,
    /// The starting byte offset of `val` from `root_val`.
    /// If the type does not have a well-defined memory layout, this is null.
    byte_offset: usize,
};

const ComptimePtrLoadKit = struct {
    /// The Value and Type corresponding to the pointee of the provided pointer.
    /// If a direct dereference is not possible, this is null.
    pointee: ?TypedValue,
    /// The largest parent Value containing `pointee` and having a well-defined memory layout.
    /// This is used for bitcasting, if direct dereferencing failed (i.e. `pointee` is null).
    parent: ?TypedValueAndOffset,
    /// Whether the `pointee` could be mutated by further
    /// semantic analysis and a copy must be performed.
    is_mutable: bool,
    /// If the root decl could not be used as `parent`, this is the type that
    /// caused that by not having a well-defined layout
    ty_without_well_defined_layout: ?Type,
};

const ComptimePtrLoadError = CompileError || error{
    RuntimeLoad,
};

/// If `maybe_array_ty` is provided, it will be used to directly dereference an
/// .elem_ptr of type T to a value of [N]T, if necessary.
fn beginComptimePtrLoad(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr_val: Value,
    maybe_array_ty: ?Type,
) ComptimePtrLoadError!ComptimePtrLoadKit {
    const target = sema.mod.getTarget();
    var deref: ComptimePtrLoadKit = switch (ptr_val.tag()) {
        .decl_ref,
        .decl_ref_mut,
        => blk: {
            const decl = switch (ptr_val.tag()) {
                .decl_ref => ptr_val.castTag(.decl_ref).?.data,
                .decl_ref_mut => ptr_val.castTag(.decl_ref_mut).?.data.decl,
                else => unreachable,
            };
            const is_mutable = ptr_val.tag() == .decl_ref_mut;
            const decl_tv = try decl.typedValue();
            if (decl_tv.val.tag() == .variable) return error.RuntimeLoad;

            const layout_defined = decl.ty.hasWellDefinedLayout();
            break :blk ComptimePtrLoadKit{
                .parent = if (layout_defined) .{ .tv = decl_tv, .byte_offset = 0 } else null,
                .pointee = decl_tv,
                .is_mutable = is_mutable,
                .ty_without_well_defined_layout = if (!layout_defined) decl.ty else null,
            };
        },

        .elem_ptr => blk: {
            const elem_ptr = ptr_val.castTag(.elem_ptr).?.data;
            const elem_ty = elem_ptr.elem_ty;
            var deref = try beginComptimePtrLoad(sema, block, src, elem_ptr.array_ptr, null);

            // This code assumes that elem_ptrs have been "flattened" in order for direct dereference
            // to succeed, meaning that elem ptrs of the same elem_ty are coalesced. Here we check that
            // our parent is not an elem_ptr with the same elem_ty, since that would be "unflattened"
            if (elem_ptr.array_ptr.castTag(.elem_ptr)) |parent_elem_ptr| assert(!(parent_elem_ptr.data.elem_ty.eql(elem_ty, target)));

            if (elem_ptr.index != 0) {
                if (elem_ty.hasWellDefinedLayout()) {
                    if (deref.parent) |*parent| {
                        // Update the byte offset (in-place)
                        const elem_size = try sema.typeAbiSize(block, src, elem_ty);
                        const offset = parent.byte_offset + elem_size * elem_ptr.index;
                        parent.byte_offset = try sema.usizeCast(block, src, offset);
                    }
                } else {
                    deref.parent = null;
                    deref.ty_without_well_defined_layout = elem_ty;
                }
            }

            // If we're loading an elem_ptr that was derived from a different type
            // than the true type of the underlying decl, we cannot deref directly
            const ty_matches = if (deref.pointee != null and deref.pointee.?.ty.isArrayOrVector()) x: {
                const deref_elem_ty = deref.pointee.?.ty.childType();
                break :x (try sema.coerceInMemoryAllowed(block, deref_elem_ty, elem_ty, false, target, src, src)) == .ok or
                    (try sema.coerceInMemoryAllowed(block, elem_ty, deref_elem_ty, false, target, src, src)) == .ok;
            } else false;
            if (!ty_matches) {
                deref.pointee = null;
                break :blk deref;
            }

            var array_tv = deref.pointee.?;
            const check_len = array_tv.ty.arrayLenIncludingSentinel();
            if (maybe_array_ty) |load_ty| {
                // It's possible that we're loading a [N]T, in which case we'd like to slice
                // the pointee array directly from our parent array.
                if (load_ty.isArrayOrVector() and load_ty.childType().eql(elem_ty, target)) {
                    const N = try sema.usizeCast(block, src, load_ty.arrayLenIncludingSentinel());
                    deref.pointee = if (elem_ptr.index + N <= check_len) TypedValue{
                        .ty = try Type.array(sema.arena, N, null, elem_ty, target),
                        .val = try array_tv.val.sliceArray(sema.arena, elem_ptr.index, elem_ptr.index + N),
                    } else null;
                    break :blk deref;
                }
            }

            deref.pointee = if (elem_ptr.index < check_len) TypedValue{
                .ty = elem_ty,
                .val = try array_tv.val.elemValue(sema.arena, elem_ptr.index),
            } else null;
            break :blk deref;
        },

        .field_ptr => blk: {
            const field_ptr = ptr_val.castTag(.field_ptr).?.data;
            const field_index = @intCast(u32, field_ptr.field_index);
            const field_ty = field_ptr.container_ty.structFieldType(field_index);
            var deref = try beginComptimePtrLoad(sema, block, src, field_ptr.container_ptr, field_ptr.container_ty);

            if (field_ptr.container_ty.hasWellDefinedLayout()) {
                if (deref.parent) |*parent| {
                    // Update the byte offset (in-place)
                    try sema.resolveTypeLayout(block, src, field_ptr.container_ty);
                    const field_offset = field_ptr.container_ty.structFieldOffset(field_index, target);
                    parent.byte_offset = try sema.usizeCast(block, src, parent.byte_offset + field_offset);
                }
            } else {
                deref.parent = null;
                deref.ty_without_well_defined_layout = field_ptr.container_ty;
            }

            if (deref.pointee) |*tv| {
                const coerce_in_mem_ok =
                    (try sema.coerceInMemoryAllowed(block, field_ptr.container_ty, tv.ty, false, target, src, src)) == .ok or
                    (try sema.coerceInMemoryAllowed(block, tv.ty, field_ptr.container_ty, false, target, src, src)) == .ok;
                if (coerce_in_mem_ok) {
                    deref.pointee = TypedValue{
                        .ty = field_ty,
                        .val = tv.val.fieldValue(tv.ty, field_index),
                    };
                    break :blk deref;
                }
            }
            deref.pointee = null;
            break :blk deref;
        },

        .opt_payload_ptr,
        .eu_payload_ptr,
        => blk: {
            const payload_ptr = ptr_val.cast(Value.Payload.PayloadPtr).?.data;
            const payload_ty = switch (ptr_val.tag()) {
                .eu_payload_ptr => payload_ptr.container_ty.errorUnionPayload(),
                .opt_payload_ptr => try payload_ptr.container_ty.optionalChildAlloc(sema.arena),
                else => unreachable,
            };
            var deref = try beginComptimePtrLoad(sema, block, src, payload_ptr.container_ptr, payload_ptr.container_ty);

            // eu_payload_ptr and opt_payload_ptr never have a well-defined layout
            if (deref.parent != null) {
                deref.parent = null;
                deref.ty_without_well_defined_layout = payload_ptr.container_ty;
            }

            if (deref.pointee) |*tv| {
                const coerce_in_mem_ok =
                    (try sema.coerceInMemoryAllowed(block, payload_ptr.container_ty, tv.ty, false, target, src, src)) == .ok or
                    (try sema.coerceInMemoryAllowed(block, tv.ty, payload_ptr.container_ty, false, target, src, src)) == .ok;
                if (coerce_in_mem_ok) {
                    const payload_val = switch (ptr_val.tag()) {
                        .eu_payload_ptr => tv.val.castTag(.eu_payload).?.data,
                        .opt_payload_ptr => tv.val.castTag(.opt_payload).?.data,
                        else => unreachable,
                    };
                    tv.* = TypedValue{ .ty = payload_ty, .val = payload_val };
                    break :blk deref;
                }
            }
            deref.pointee = null;
            break :blk deref;
        },

        .zero,
        .one,
        .int_u64,
        .int_i64,
        .int_big_positive,
        .int_big_negative,
        .variable,
        .extern_fn,
        .function,
        => return error.RuntimeLoad,

        else => unreachable,
    };

    if (deref.pointee) |tv| {
        if (deref.parent == null and tv.ty.hasWellDefinedLayout()) {
            deref.parent = .{ .tv = tv, .byte_offset = 0 };
        }
    }
    return deref;
}

fn bitCast(
    sema: *Sema,
    block: *Block,
    dest_ty_unresolved: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const dest_ty = try sema.resolveTypeFields(block, inst_src, dest_ty_unresolved);
    try sema.resolveTypeLayout(block, inst_src, dest_ty);

    const old_ty = try sema.resolveTypeFields(block, inst_src, sema.typeOf(inst));
    try sema.resolveTypeLayout(block, inst_src, old_ty);

    const target = sema.mod.getTarget();
    var dest_bits = dest_ty.bitSize(target);
    var old_bits = old_ty.bitSize(target);

    if (old_bits != dest_bits) {
        return sema.fail(block, inst_src, "@bitCast size mismatch: destination type '{}' has {d} bits but source type '{}' has {d} bits", .{
            dest_ty.fmt(target),
            dest_bits,
            old_ty.fmt(target),
            old_bits,
        });
    }

    if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |val| {
        const result_val = try sema.bitCastVal(block, inst_src, val, old_ty, dest_ty, 0);
        return sema.addConstant(dest_ty, result_val);
    }
    try sema.requireRuntimeBlock(block, inst_src);
    return block.addBitCast(dest_ty, inst);
}

pub fn bitCastVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    val: Value,
    old_ty: Type,
    new_ty: Type,
    buffer_offset: usize,
) !Value {
    const target = sema.mod.getTarget();
    if (old_ty.eql(new_ty, target)) return val;

    // For types with well-defined memory layouts, we serialize them a byte buffer,
    // then deserialize to the new type.
    const abi_size = try sema.usizeCast(block, src, old_ty.abiSize(target));
    const buffer = try sema.gpa.alloc(u8, abi_size);
    defer sema.gpa.free(buffer);
    val.writeToMemory(old_ty, sema.mod, buffer);
    return Value.readFromMemory(new_ty, sema.mod, buffer[buffer_offset..], sema.arena);
}

fn coerceArrayPtrToSlice(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |val| {
        const ptr_array_ty = sema.typeOf(inst);
        const array_ty = ptr_array_ty.childType();
        const slice_val = try Value.Tag.slice.create(sema.arena, .{
            .ptr = val,
            .len = try Value.Tag.int_u64.create(sema.arena, array_ty.arrayLen()),
        });
        return sema.addConstant(dest_ty, slice_val);
    }
    try sema.requireRuntimeBlock(block, inst_src);
    return block.addTyOp(.array_to_slice, dest_ty, inst);
}

fn coerceCompatiblePtrs(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    // TODO check const/volatile/alignment
    if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |val| {
        // The comptime Value representation is compatible with both types.
        return sema.addConstant(dest_ty, val);
    }
    try sema.requireRuntimeBlock(block, inst_src);
    return sema.bitCast(block, dest_ty, inst, inst_src);
}

fn coerceEnumToUnion(
    sema: *Sema,
    block: *Block,
    union_ty: Type,
    union_ty_src: LazySrcLoc,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    const target = sema.mod.getTarget();

    const tag_ty = union_ty.unionTagType() orelse {
        const msg = msg: {
            const msg = try sema.errMsg(block, inst_src, "expected {}, found {}", .{
                union_ty.fmt(target), inst_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(block, union_ty_src, msg, "cannot coerce enum to untagged union", .{});
            try sema.addDeclaredHereNote(msg, union_ty);
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    };

    const enum_tag = try sema.coerce(block, tag_ty, inst, inst_src);
    if (try sema.resolveDefinedValue(block, inst_src, enum_tag)) |val| {
        const union_obj = union_ty.cast(Type.Payload.Union).?.data;
        const field_index = union_obj.tag_ty.enumTagFieldIndex(val, target) orelse {
            const msg = msg: {
                const msg = try sema.errMsg(block, inst_src, "union {} has no tag with value {}", .{
                    union_ty.fmt(target), val.fmtValue(tag_ty, target),
                });
                errdefer msg.destroy(sema.gpa);
                try sema.addDeclaredHereNote(msg, union_ty);
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        };
        const field = union_obj.fields.values()[field_index];
        const field_ty = try sema.resolveTypeFields(block, inst_src, field.ty);
        const opv = (try sema.typeHasOnePossibleValue(block, inst_src, field_ty)) orelse {
            // TODO resolve the field names and include in the error message,
            // also instead of 'union declared here' make it 'field "foo" declared here'.
            const msg = msg: {
                const msg = try sema.errMsg(block, inst_src, "coercion to union {} must initialize {} field", .{
                    union_ty.fmt(target), field_ty.fmt(target),
                });
                errdefer msg.destroy(sema.gpa);
                try sema.addDeclaredHereNote(msg, union_ty);
                break :msg msg;
            };
            return sema.failWithOwnedErrorMsg(block, msg);
        };

        return sema.addConstant(union_ty, try Value.Tag.@"union".create(sema.arena, .{
            .tag = val,
            .val = opv,
        }));
    }

    try sema.requireRuntimeBlock(block, inst_src);

    if (tag_ty.isNonexhaustiveEnum()) {
        const msg = msg: {
            const msg = try sema.errMsg(block, inst_src, "runtime coercion to union {} from non-exhaustive enum", .{
                union_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);
            try sema.addDeclaredHereNote(msg, tag_ty);
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    // If the union has all fields 0 bits, the union value is just the enum value.
    if (union_ty.unionHasAllZeroBitFieldTypes()) {
        return block.addBitCast(union_ty, enum_tag);
    }

    // TODO resolve the field names and add a hint that says "field 'foo' has type 'bar'"
    // instead of the "union declared here" hint
    const msg = msg: {
        const msg = try sema.errMsg(block, inst_src, "runtime coercion to union {} which has non-void fields", .{
            union_ty.fmt(target),
        });
        errdefer msg.destroy(sema.gpa);
        try sema.addDeclaredHereNote(msg, union_ty);
        break :msg msg;
    };
    return sema.failWithOwnedErrorMsg(block, msg);
}

fn coerceAnonStructToUnion(
    sema: *Sema,
    block: *Block,
    union_ty: Type,
    union_ty_src: LazySrcLoc,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    const anon_struct = inst_ty.castTag(.anon_struct).?.data;
    if (anon_struct.types.len != 1) {
        const msg = msg: {
            const msg = try sema.errMsg(
                block,
                inst_src,
                "cannot initialize multiple union fields at once, unions can only have one active field",
                .{},
            );
            errdefer msg.destroy(sema.gpa);

            // TODO add notes for where the anon struct was created to point out
            // the extra fields.

            try sema.addDeclaredHereNote(msg, union_ty);
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const field_name = anon_struct.names[0];
    const init = try sema.structFieldVal(block, inst_src, inst, field_name, inst_src, inst_ty);
    return sema.unionInit(block, init, inst_src, union_ty, union_ty_src, field_name, inst_src);
}

fn coerceAnonStructToUnionPtrs(
    sema: *Sema,
    block: *Block,
    ptr_union_ty: Type,
    union_ty_src: LazySrcLoc,
    ptr_anon_struct: Air.Inst.Ref,
    anon_struct_src: LazySrcLoc,
) !Air.Inst.Ref {
    const union_ty = ptr_union_ty.childType();
    const anon_struct = try sema.analyzeLoad(block, anon_struct_src, ptr_anon_struct, anon_struct_src);
    const union_inst = try sema.coerceAnonStructToUnion(block, union_ty, union_ty_src, anon_struct, anon_struct_src);
    return sema.analyzeRef(block, union_ty_src, union_inst);
}

fn coerceAnonStructToStructPtrs(
    sema: *Sema,
    block: *Block,
    ptr_struct_ty: Type,
    struct_ty_src: LazySrcLoc,
    ptr_anon_struct: Air.Inst.Ref,
    anon_struct_src: LazySrcLoc,
) !Air.Inst.Ref {
    const struct_ty = ptr_struct_ty.childType();
    const anon_struct = try sema.analyzeLoad(block, anon_struct_src, ptr_anon_struct, anon_struct_src);
    const struct_inst = try sema.coerceTupleToStruct(block, struct_ty, struct_ty_src, anon_struct, anon_struct_src);
    return sema.analyzeRef(block, struct_ty_src, struct_inst);
}

/// If the lengths match, coerces element-wise.
fn coerceArrayLike(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    dest_ty_src: LazySrcLoc,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    const inst_len = inst_ty.arrayLen();
    const dest_len = try sema.usizeCast(block, dest_ty_src, dest_ty.arrayLen());
    const target = sema.mod.getTarget();

    if (dest_len != inst_len) {
        const msg = msg: {
            const msg = try sema.errMsg(block, inst_src, "expected {}, found {}", .{
                dest_ty.fmt(target), inst_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(block, dest_ty_src, msg, "destination has length {d}", .{dest_len});
            try sema.errNote(block, inst_src, msg, "source has length {d}", .{inst_len});
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const dest_elem_ty = dest_ty.childType();
    const inst_elem_ty = inst_ty.childType();
    const in_memory_result = try sema.coerceInMemoryAllowed(block, dest_elem_ty, inst_elem_ty, false, target, dest_ty_src, inst_src);
    if (in_memory_result == .ok) {
        if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |inst_val| {
            // These types share the same comptime value representation.
            return sema.addConstant(dest_ty, inst_val);
        }
        try sema.requireRuntimeBlock(block, inst_src);
        return block.addBitCast(dest_ty, inst);
    }

    const element_vals = try sema.arena.alloc(Value, dest_len);
    const element_refs = try sema.arena.alloc(Air.Inst.Ref, dest_len);
    var runtime_src: ?LazySrcLoc = null;

    for (element_vals) |*elem, i| {
        const index_ref = try sema.addConstant(
            Type.usize,
            try Value.Tag.int_u64.create(sema.arena, i),
        );
        const elem_src = inst_src; // TODO better source location
        const elem_ref = try elemValArray(sema, block, inst_src, inst, elem_src, index_ref);
        const coerced = try sema.coerce(block, dest_elem_ty, elem_ref, elem_src);
        element_refs[i] = coerced;
        if (runtime_src == null) {
            if (try sema.resolveMaybeUndefVal(block, elem_src, coerced)) |elem_val| {
                elem.* = elem_val;
            } else {
                runtime_src = elem_src;
            }
        }
    }

    if (runtime_src) |rs| {
        try sema.requireRuntimeBlock(block, rs);
        return block.addAggregateInit(dest_ty, element_refs);
    }

    return sema.addConstant(
        dest_ty,
        try Value.Tag.aggregate.create(sema.arena, element_vals),
    );
}

/// If the lengths match, coerces element-wise.
fn coerceTupleToArray(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    dest_ty_src: LazySrcLoc,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    const inst_len = inst_ty.arrayLen();
    const dest_len = try sema.usizeCast(block, dest_ty_src, dest_ty.arrayLen());
    const target = sema.mod.getTarget();

    if (dest_len != inst_len) {
        const msg = msg: {
            const msg = try sema.errMsg(block, inst_src, "expected {}, found {}", .{
                dest_ty.fmt(target), inst_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);
            try sema.errNote(block, dest_ty_src, msg, "destination has length {d}", .{dest_len});
            try sema.errNote(block, inst_src, msg, "source has length {d}", .{inst_len});
            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const element_vals = try sema.arena.alloc(Value, dest_len);
    const element_refs = try sema.arena.alloc(Air.Inst.Ref, dest_len);
    const dest_elem_ty = dest_ty.childType();

    var runtime_src: ?LazySrcLoc = null;
    for (element_vals) |*elem, i_usize| {
        const i = @intCast(u32, i_usize);
        const elem_src = inst_src; // TODO better source location
        const elem_ref = try tupleField(sema, block, inst_src, inst, elem_src, i);
        const coerced = try sema.coerce(block, dest_elem_ty, elem_ref, elem_src);
        element_refs[i] = coerced;
        if (runtime_src == null) {
            if (try sema.resolveMaybeUndefVal(block, elem_src, coerced)) |elem_val| {
                elem.* = elem_val;
            } else {
                runtime_src = elem_src;
            }
        }
    }

    if (runtime_src) |rs| {
        try sema.requireRuntimeBlock(block, rs);
        return block.addAggregateInit(dest_ty, element_refs);
    }

    return sema.addConstant(
        dest_ty,
        try Value.Tag.aggregate.create(sema.arena, element_vals),
    );
}

/// If the lengths match, coerces element-wise.
fn coerceTupleToSlicePtrs(
    sema: *Sema,
    block: *Block,
    slice_ty: Type,
    slice_ty_src: LazySrcLoc,
    ptr_tuple: Air.Inst.Ref,
    tuple_src: LazySrcLoc,
) !Air.Inst.Ref {
    const tuple_ty = sema.typeOf(ptr_tuple).childType();
    const tuple = try sema.analyzeLoad(block, tuple_src, ptr_tuple, tuple_src);
    const slice_info = slice_ty.ptrInfo().data;
    const target = sema.mod.getTarget();
    const array_ty = try Type.array(sema.arena, tuple_ty.structFieldCount(), slice_info.sentinel, slice_info.pointee_type, target);
    const array_inst = try sema.coerceTupleToArray(block, array_ty, slice_ty_src, tuple, tuple_src);
    if (slice_info.@"align" != 0) {
        return sema.fail(block, slice_ty_src, "TODO: override the alignment of the array decl we create here", .{});
    }
    const ptr_array = try sema.analyzeRef(block, slice_ty_src, array_inst);
    return sema.coerceArrayPtrToSlice(block, slice_ty, ptr_array, slice_ty_src);
}

/// If the lengths match, coerces element-wise.
fn coerceTupleToArrayPtrs(
    sema: *Sema,
    block: *Block,
    ptr_array_ty: Type,
    array_ty_src: LazySrcLoc,
    ptr_tuple: Air.Inst.Ref,
    tuple_src: LazySrcLoc,
) !Air.Inst.Ref {
    const tuple = try sema.analyzeLoad(block, tuple_src, ptr_tuple, tuple_src);
    const ptr_info = ptr_array_ty.ptrInfo().data;
    const array_ty = ptr_info.pointee_type;
    const array_inst = try sema.coerceTupleToArray(block, array_ty, array_ty_src, tuple, tuple_src);
    if (ptr_info.@"align" != 0) {
        return sema.fail(block, array_ty_src, "TODO: override the alignment of the array decl we create here", .{});
    }
    const ptr_array = try sema.analyzeRef(block, array_ty_src, array_inst);
    return ptr_array;
}

/// Handles both tuples and anon struct literals. Coerces field-wise. Reports
/// errors for both extra fields and missing fields.
fn coerceTupleToStruct(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    dest_ty_src: LazySrcLoc,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const struct_ty = try sema.resolveTypeFields(block, dest_ty_src, dest_ty);

    if (struct_ty.isTupleOrAnonStruct()) {
        return sema.fail(block, dest_ty_src, "TODO: implement coercion from tuples to tuples", .{});
    }

    const fields = struct_ty.structFields();
    const field_vals = try sema.arena.alloc(Value, fields.count());
    const field_refs = try sema.arena.alloc(Air.Inst.Ref, field_vals.len);
    mem.set(Air.Inst.Ref, field_refs, .none);

    const inst_ty = sema.typeOf(inst);
    const tuple = inst_ty.tupleFields();
    var runtime_src: ?LazySrcLoc = null;
    for (tuple.types) |_, i_usize| {
        const i = @intCast(u32, i_usize);
        const field_src = inst_src; // TODO better source location
        const field_name = if (inst_ty.castTag(.anon_struct)) |payload|
            payload.data.names[i]
        else
            try std.fmt.allocPrint(sema.arena, "{d}", .{i});
        const field_index = try sema.structFieldIndex(block, struct_ty, field_name, field_src);
        const field = fields.values()[field_index];
        if (field.is_comptime) {
            return sema.fail(block, dest_ty_src, "TODO: implement coercion from tuples to structs when one of the destination struct fields is comptime", .{});
        }
        const elem_ref = try tupleField(sema, block, inst_src, inst, field_src, i);
        const coerced = try sema.coerce(block, field.ty, elem_ref, field_src);
        field_refs[field_index] = coerced;
        if (runtime_src == null) {
            if (try sema.resolveMaybeUndefVal(block, field_src, coerced)) |field_val| {
                field_vals[field_index] = field_val;
            } else {
                runtime_src = field_src;
            }
        }
    }

    // Populate default field values and report errors for missing fields.
    var root_msg: ?*Module.ErrorMsg = null;

    for (field_refs) |*field_ref, i| {
        if (field_ref.* != .none) continue;

        const field_name = fields.keys()[i];
        const field = fields.values()[i];
        const field_src = inst_src; // TODO better source location
        if (field.default_val.tag() == .unreachable_value) {
            const template = "missing struct field: {s}";
            const args = .{field_name};
            if (root_msg) |msg| {
                try sema.errNote(block, field_src, msg, template, args);
            } else {
                root_msg = try sema.errMsg(block, field_src, template, args);
            }
            continue;
        }
        if (runtime_src == null) {
            field_vals[i] = field.default_val;
        } else {
            field_ref.* = try sema.addConstant(field.ty, field.default_val);
        }
    }

    if (root_msg) |msg| {
        try sema.addDeclaredHereNote(msg, struct_ty);
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    if (runtime_src) |rs| {
        try sema.requireRuntimeBlock(block, rs);
        return block.addAggregateInit(struct_ty, field_refs);
    }

    return sema.addConstant(
        struct_ty,
        try Value.Tag.aggregate.create(sema.arena, field_vals),
    );
}

fn analyzeDeclVal(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    decl: *Decl,
) CompileError!Air.Inst.Ref {
    if (sema.decl_val_table.get(decl)) |result| {
        return result;
    }
    const decl_ref = try sema.analyzeDeclRef(decl);
    const result = try sema.analyzeLoad(block, src, decl_ref, src);
    if (Air.refToIndex(result)) |index| {
        if (sema.air_instructions.items(.tag)[index] == .constant) {
            try sema.decl_val_table.put(sema.gpa, decl, result);
        }
    }
    return result;
}

fn ensureDeclAnalyzed(sema: *Sema, decl: *Decl) CompileError!void {
    sema.mod.ensureDeclAnalyzed(decl) catch |err| {
        if (sema.owner_func) |owner_func| {
            owner_func.state = .dependency_failure;
        } else {
            sema.owner_decl.analysis = .dependency_failure;
        }
        return err;
    };
}

fn ensureFuncBodyAnalyzed(sema: *Sema, func: *Module.Fn) CompileError!void {
    sema.mod.ensureFuncBodyAnalyzed(func) catch |err| {
        if (sema.owner_func) |owner_func| {
            owner_func.state = .dependency_failure;
        } else {
            sema.owner_decl.analysis = .dependency_failure;
        }
        return err;
    };
}

fn refValue(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type, val: Value) !Value {
    var anon_decl = try block.startAnonDecl(src);
    defer anon_decl.deinit();
    const decl = try anon_decl.finish(
        try ty.copy(anon_decl.arena()),
        try val.copy(anon_decl.arena()),
        0, // default alignment
    );
    try sema.mod.declareDeclDependency(sema.owner_decl, decl);
    return try Value.Tag.decl_ref.create(sema.arena, decl);
}

fn optRefValue(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type, opt_val: ?Value) !Value {
    const val = opt_val orelse return Value.@"null";
    const ptr_val = try refValue(sema, block, src, ty, val);
    const result = try Value.Tag.opt_payload.create(sema.arena, ptr_val);
    return result;
}

fn analyzeDeclRef(sema: *Sema, decl: *Decl) CompileError!Air.Inst.Ref {
    try sema.mod.declareDeclDependency(sema.owner_decl, decl);
    try sema.ensureDeclAnalyzed(decl);

    const target = sema.mod.getTarget();
    const decl_tv = try decl.typedValue();
    if (decl_tv.val.castTag(.variable)) |payload| {
        const variable = payload.data;
        const ty = try Type.ptr(sema.arena, target, .{
            .pointee_type = decl_tv.ty,
            .mutable = variable.is_mutable,
            .@"addrspace" = decl.@"addrspace",
            .@"align" = decl.@"align",
        });
        return sema.addConstant(ty, try Value.Tag.decl_ref.create(sema.arena, decl));
    }
    return sema.addConstant(
        try Type.ptr(sema.arena, target, .{
            .pointee_type = decl_tv.ty,
            .mutable = false,
            .@"addrspace" = decl.@"addrspace",
        }),
        try Value.Tag.decl_ref.create(sema.arena, decl),
    );
}

fn analyzeRef(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    operand: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const operand_ty = sema.typeOf(operand);

    if (try sema.resolveMaybeUndefVal(block, src, operand)) |val| {
        var anon_decl = try block.startAnonDecl(src);
        defer anon_decl.deinit();
        return sema.analyzeDeclRef(try anon_decl.finish(
            try operand_ty.copy(anon_decl.arena()),
            try val.copy(anon_decl.arena()),
            0, // default alignment
        ));
    }

    try sema.requireRuntimeBlock(block, src);
    const address_space = target_util.defaultAddressSpace(sema.mod.getTarget(), .local);
    const target = sema.mod.getTarget();
    const ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = operand_ty,
        .mutable = false,
        .@"addrspace" = address_space,
    });
    const mut_ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = operand_ty,
        .@"addrspace" = address_space,
    });
    const alloc = try block.addTy(.alloc, mut_ptr_type);
    try sema.storePtr(block, src, alloc, operand);

    // TODO: Replace with sema.coerce when that supports adding pointer constness.
    return sema.bitCast(block, ptr_type, alloc, src);
}

fn analyzeLoad(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr: Air.Inst.Ref,
    ptr_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const target = sema.mod.getTarget();
    const ptr_ty = sema.typeOf(ptr);
    const elem_ty = switch (ptr_ty.zigTypeTag()) {
        .Pointer => ptr_ty.childType(),
        else => return sema.fail(block, ptr_src, "expected pointer, found '{}'", .{ptr_ty.fmt(target)}),
    };
    if (try sema.resolveDefinedValue(block, ptr_src, ptr)) |ptr_val| {
        if (try sema.pointerDeref(block, ptr_src, ptr_val, ptr_ty)) |elem_val| {
            return sema.addConstant(elem_ty, elem_val);
        }
    }

    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.load, elem_ty, ptr);
}

fn analyzeSlicePtr(
    sema: *Sema,
    block: *Block,
    slice_src: LazySrcLoc,
    slice: Air.Inst.Ref,
    slice_ty: Type,
) CompileError!Air.Inst.Ref {
    const buf = try sema.arena.create(Type.SlicePtrFieldTypeBuffer);
    const result_ty = slice_ty.slicePtrFieldType(buf);
    if (try sema.resolveMaybeUndefVal(block, slice_src, slice)) |val| {
        if (val.isUndef()) return sema.addConstUndef(result_ty);
        return sema.addConstant(result_ty, val.slicePtr());
    }
    try sema.requireRuntimeBlock(block, slice_src);
    return block.addTyOp(.slice_ptr, result_ty, slice);
}

fn analyzeSliceLen(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    slice_inst: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    if (try sema.resolveMaybeUndefVal(block, src, slice_inst)) |slice_val| {
        if (slice_val.isUndef()) {
            return sema.addConstUndef(Type.usize);
        }
        const target = sema.mod.getTarget();
        return sema.addIntUnsigned(Type.usize, slice_val.sliceLen(target));
    }
    try sema.requireRuntimeBlock(block, src);
    return block.addTyOp(.slice_len, Type.usize, slice_inst);
}

fn analyzeIsNull(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    operand: Air.Inst.Ref,
    invert_logic: bool,
) CompileError!Air.Inst.Ref {
    const result_ty = Type.bool;
    if (try sema.resolveMaybeUndefVal(block, src, operand)) |opt_val| {
        if (opt_val.isUndef()) {
            return sema.addConstUndef(result_ty);
        }
        const is_null = opt_val.isNull();
        const bool_value = if (invert_logic) !is_null else is_null;
        if (bool_value) {
            return Air.Inst.Ref.bool_true;
        } else {
            return Air.Inst.Ref.bool_false;
        }
    }
    try sema.requireRuntimeBlock(block, src);
    const air_tag: Air.Inst.Tag = if (invert_logic) .is_non_null else .is_null;
    return block.addUnOp(air_tag, operand);
}

fn analyzeIsNonErr(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    operand: Air.Inst.Ref,
) CompileError!Air.Inst.Ref {
    const operand_ty = sema.typeOf(operand);
    const ot = operand_ty.zigTypeTag();
    if (ot != .ErrorSet and ot != .ErrorUnion) return Air.Inst.Ref.bool_true;
    if (ot == .ErrorSet) return Air.Inst.Ref.bool_false;
    assert(ot == .ErrorUnion);

    if (Air.refToIndex(operand)) |operand_inst| {
        const air_tags = sema.air_instructions.items(.tag);
        if (air_tags[operand_inst] == .wrap_errunion_payload) {
            return Air.Inst.Ref.bool_true;
        }
    }

    const maybe_operand_val = try sema.resolveMaybeUndefVal(block, src, operand);

    // exception if the error union error set is known to be empty,
    // we allow the comparison but always make it comptime known.
    const set_ty = operand_ty.errorUnionSet();
    switch (set_ty.tag()) {
        .anyerror => {},
        .error_set_inferred => blk: {
            // If the error set is empty, we must return a comptime true or false.
            // However we want to avoid unnecessarily resolving an inferred error set
            // in case it is already non-empty.
            const ies = set_ty.castTag(.error_set_inferred).?.data;
            if (ies.is_anyerror) break :blk;
            if (ies.errors.count() != 0) break :blk;
            if (maybe_operand_val == null) {
                try sema.resolveInferredErrorSet(block, src, ies);
                if (ies.is_anyerror) break :blk;
                if (ies.errors.count() == 0) return Air.Inst.Ref.bool_true;
            }
        },
        else => if (set_ty.errorSetNames().len == 0) return Air.Inst.Ref.bool_true,
    }

    if (maybe_operand_val) |err_union| {
        if (err_union.isUndef()) {
            return sema.addConstUndef(Type.bool);
        }
        if (err_union.getError() == null) {
            return Air.Inst.Ref.bool_true;
        } else {
            return Air.Inst.Ref.bool_false;
        }
    }
    try sema.requireRuntimeBlock(block, src);
    return block.addUnOp(.is_non_err, operand);
}

fn analyzeSlice(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ptr_ptr: Air.Inst.Ref,
    uncasted_start: Air.Inst.Ref,
    uncasted_end_opt: Air.Inst.Ref,
    sentinel_opt: Air.Inst.Ref,
    sentinel_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const ptr_src = src; // TODO better source location
    const start_src = src; // TODO better source location
    const end_src = src; // TODO better source location
    // Slice expressions can operate on a variable whose type is an array. This requires
    // the slice operand to be a pointer. In the case of a non-array, it will be a double pointer.
    const ptr_ptr_ty = sema.typeOf(ptr_ptr);
    const target = sema.mod.getTarget();
    const ptr_ptr_child_ty = switch (ptr_ptr_ty.zigTypeTag()) {
        .Pointer => ptr_ptr_ty.elemType(),
        else => return sema.fail(block, ptr_src, "expected pointer, found '{}'", .{ptr_ptr_ty.fmt(target)}),
    };

    var array_ty = ptr_ptr_child_ty;
    var slice_ty = ptr_ptr_ty;
    var ptr_or_slice = ptr_ptr;
    var elem_ty = ptr_ptr_child_ty.childType();
    var ptr_sentinel: ?Value = null;
    switch (ptr_ptr_child_ty.zigTypeTag()) {
        .Array => {
            ptr_sentinel = ptr_ptr_child_ty.sentinel();
        },
        .Pointer => switch (ptr_ptr_child_ty.ptrSize()) {
            .One => {
                const double_child_ty = ptr_ptr_child_ty.childType();
                if (double_child_ty.zigTypeTag() == .Array) {
                    ptr_sentinel = double_child_ty.sentinel();
                    ptr_or_slice = try sema.analyzeLoad(block, src, ptr_ptr, ptr_src);
                    slice_ty = ptr_ptr_child_ty;
                    array_ty = double_child_ty;
                    elem_ty = double_child_ty.childType();
                } else {
                    return sema.fail(block, ptr_src, "slice of single-item pointer", .{});
                }
            },
            .Many, .C => {
                ptr_sentinel = ptr_ptr_child_ty.sentinel();
                ptr_or_slice = try sema.analyzeLoad(block, src, ptr_ptr, ptr_src);
                slice_ty = ptr_ptr_child_ty;
                array_ty = ptr_ptr_child_ty;
                elem_ty = ptr_ptr_child_ty.childType();

                if (ptr_ptr_child_ty.ptrSize() == .C) {
                    if (try sema.resolveDefinedValue(block, ptr_src, ptr_or_slice)) |ptr_val| {
                        if (ptr_val.isNull()) {
                            return sema.fail(block, ptr_src, "slice of null pointer", .{});
                        }
                    }
                }
            },
            .Slice => {
                ptr_sentinel = ptr_ptr_child_ty.sentinel();
                ptr_or_slice = try sema.analyzeLoad(block, src, ptr_ptr, ptr_src);
                slice_ty = ptr_ptr_child_ty;
                array_ty = ptr_ptr_child_ty;
                elem_ty = ptr_ptr_child_ty.childType();
            },
        },
        else => return sema.fail(block, ptr_src, "slice of non-array type '{}'", .{ptr_ptr_child_ty.fmt(target)}),
    }

    const ptr = if (slice_ty.isSlice())
        try sema.analyzeSlicePtr(block, ptr_src, ptr_or_slice, slice_ty)
    else
        ptr_or_slice;

    const start = try sema.coerce(block, Type.usize, uncasted_start, start_src);
    const new_ptr = try analyzePtrArithmetic(sema, block, src, ptr, start, .ptr_add, ptr_src, start_src);

    // true if and only if the end index of the slice, implicitly or explicitly, equals
    // the length of the underlying object being sliced. we might learn the length of the
    // underlying object because it is an array (which has the length in the type), or
    // we might learn of the length because it is a comptime-known slice value.
    var end_is_len = uncasted_end_opt == .none;
    const end = e: {
        if (array_ty.zigTypeTag() == .Array) {
            const len_val = try Value.Tag.int_u64.create(sema.arena, array_ty.arrayLen());

            if (!end_is_len) {
                const end = try sema.coerce(block, Type.usize, uncasted_end_opt, end_src);
                if (try sema.resolveMaybeUndefVal(block, end_src, end)) |end_val| {
                    const len_s_val = try Value.Tag.int_u64.create(
                        sema.arena,
                        array_ty.arrayLenIncludingSentinel(),
                    );
                    if (end_val.compare(.gt, len_s_val, Type.usize, target)) {
                        const sentinel_label: []const u8 = if (array_ty.sentinel() != null)
                            " +1 (sentinel)"
                        else
                            "";

                        return sema.fail(
                            block,
                            end_src,
                            "end index {} out of bounds for array of length {}{s}",
                            .{
                                end_val.fmtValue(Type.usize, target),
                                len_val.fmtValue(Type.usize, target),
                                sentinel_label,
                            },
                        );
                    }

                    // end_is_len is only true if we are NOT using the sentinel
                    // length. For sentinel-length, we don't want the type to
                    // contain the sentinel.
                    if (end_val.eql(len_val, Type.usize, target)) {
                        end_is_len = true;
                    }
                }
                break :e end;
            }

            break :e try sema.addConstant(Type.usize, len_val);
        } else if (slice_ty.isSlice()) {
            if (!end_is_len) {
                const end = try sema.coerce(block, Type.usize, uncasted_end_opt, end_src);
                if (try sema.resolveDefinedValue(block, end_src, end)) |end_val| {
                    if (try sema.resolveDefinedValue(block, src, ptr_or_slice)) |slice_val| {
                        const has_sentinel = slice_ty.sentinel() != null;
                        var int_payload: Value.Payload.U64 = .{
                            .base = .{ .tag = .int_u64 },
                            .data = slice_val.sliceLen(target) + @boolToInt(has_sentinel),
                        };
                        const slice_len_val = Value.initPayload(&int_payload.base);
                        if (end_val.compare(.gt, slice_len_val, Type.usize, target)) {
                            const sentinel_label: []const u8 = if (has_sentinel)
                                " +1 (sentinel)"
                            else
                                "";

                            return sema.fail(
                                block,
                                end_src,
                                "end index {} out of bounds for slice of length {d}{s}",
                                .{
                                    end_val.fmtValue(Type.usize, target),
                                    slice_val.sliceLen(target),
                                    sentinel_label,
                                },
                            );
                        }

                        // If the slice has a sentinel, we subtract one so that
                        // end_is_len is only true if it equals the length WITHOUT
                        // the sentinel, so we don't add a sentinel type.
                        if (has_sentinel) {
                            int_payload.data -= 1;
                        }

                        if (end_val.eql(slice_len_val, Type.usize, target)) {
                            end_is_len = true;
                        }
                    }
                }
                break :e end;
            }
            break :e try sema.analyzeSliceLen(block, src, ptr_or_slice);
        }
        if (!end_is_len) {
            break :e try sema.coerce(block, Type.usize, uncasted_end_opt, end_src);
        }
        return sema.fail(block, end_src, "slice of pointer must include end value", .{});
    };

    const sentinel = s: {
        if (sentinel_opt != .none) {
            const casted = try sema.coerce(block, elem_ty, sentinel_opt, sentinel_src);
            break :s try sema.resolveConstValue(block, sentinel_src, casted);
        }
        // If we are slicing to the end of something that is sentinel-terminated
        // then the resulting slice type is also sentinel-terminated.
        if (end_is_len) {
            if (ptr_sentinel) |sent| {
                break :s sent;
            }
        }
        break :s null;
    };

    // requirement: start <= end
    if (try sema.resolveDefinedValue(block, src, end)) |end_val| {
        if (try sema.resolveDefinedValue(block, src, start)) |start_val| {
            if (start_val.compare(.gt, end_val, Type.usize, target)) {
                return sema.fail(
                    block,
                    start_src,
                    "start index {} is larger than end index {}",
                    .{
                        start_val.fmtValue(Type.usize, target),
                        end_val.fmtValue(Type.usize, target),
                    },
                );
            }
        }
    }

    const new_len = try sema.analyzeArithmetic(block, .sub, end, start, src, end_src, start_src);
    const opt_new_len_val = try sema.resolveDefinedValue(block, src, new_len);

    const new_ptr_ty_info = sema.typeOf(new_ptr).ptrInfo().data;
    const new_allowzero = new_ptr_ty_info.@"allowzero" and sema.typeOf(ptr).ptrSize() != .C;

    if (opt_new_len_val) |new_len_val| {
        const new_len_int = new_len_val.toUnsignedInt(target);

        const return_ty = try Type.ptr(sema.arena, target, .{
            .pointee_type = try Type.array(sema.arena, new_len_int, sentinel, elem_ty, target),
            .sentinel = null,
            .@"align" = new_ptr_ty_info.@"align",
            .@"addrspace" = new_ptr_ty_info.@"addrspace",
            .mutable = new_ptr_ty_info.mutable,
            .@"allowzero" = new_allowzero,
            .@"volatile" = new_ptr_ty_info.@"volatile",
            .size = .One,
        });

        const opt_new_ptr_val = try sema.resolveMaybeUndefVal(block, ptr_src, new_ptr);
        const new_ptr_val = opt_new_ptr_val orelse {
            return block.addBitCast(return_ty, new_ptr);
        };

        if (!new_ptr_val.isUndef()) {
            return sema.addConstant(return_ty, new_ptr_val);
        }

        // Special case: @as([]i32, undefined)[x..x]
        if (new_len_int == 0) {
            return sema.addConstUndef(return_ty);
        }

        return sema.fail(block, ptr_src, "non-zero length slice of undefined pointer", .{});
    }

    const return_ty = try Type.ptr(sema.arena, target, .{
        .pointee_type = elem_ty,
        .sentinel = sentinel,
        .@"align" = new_ptr_ty_info.@"align",
        .@"addrspace" = new_ptr_ty_info.@"addrspace",
        .mutable = new_ptr_ty_info.mutable,
        .@"allowzero" = new_allowzero,
        .@"volatile" = new_ptr_ty_info.@"volatile",
        .size = .Slice,
    });

    try sema.requireRuntimeBlock(block, src);
    if (block.wantSafety()) {
        // requirement: slicing C ptr is non-null
        if (ptr_ptr_child_ty.isCPtr()) {
            const is_non_null = try sema.analyzeIsNull(block, ptr_src, ptr, true);
            try sema.addSafetyCheck(block, is_non_null, .unwrap_null);
        }

        // requirement: end <= len
        const opt_len_inst = if (array_ty.zigTypeTag() == .Array)
            try sema.addIntUnsigned(Type.usize, array_ty.arrayLenIncludingSentinel())
        else if (slice_ty.isSlice()) blk: {
            if (try sema.resolveDefinedValue(block, src, ptr_or_slice)) |slice_val| {
                // we don't need to add one for sentinels because the
                // underlying value data includes the sentinel
                break :blk try sema.addIntUnsigned(Type.usize, slice_val.sliceLen(target));
            }

            const slice_len_inst = try block.addTyOp(.slice_len, Type.usize, ptr_or_slice);
            if (slice_ty.sentinel() == null) break :blk slice_len_inst;

            // we have to add one because slice lengths don't include the sentinel
            break :blk try sema.analyzeArithmetic(block, .add, slice_len_inst, .one, src, end_src, end_src);
        } else null;
        if (opt_len_inst) |len_inst| {
            const end_is_in_bounds = try block.addBinOp(.cmp_lte, end, len_inst);
            try sema.addSafetyCheck(block, end_is_in_bounds, .index_out_of_bounds);
        }

        // requirement: start <= end
        const start_is_in_bounds = try block.addBinOp(.cmp_lte, start, end);
        try sema.addSafetyCheck(block, start_is_in_bounds, .index_out_of_bounds);
    }
    return block.addInst(.{
        .tag = .slice,
        .data = .{ .ty_pl = .{
            .ty = try sema.addType(return_ty),
            .payload = try sema.addExtra(Air.Bin{
                .lhs = new_ptr,
                .rhs = new_len,
            }),
        } },
    });
}

/// Asserts that lhs and rhs types are both numeric.
fn cmpNumeric(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
    op: std.math.CompareOperator,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);

    assert(lhs_ty.isNumeric());
    assert(rhs_ty.isNumeric());

    const lhs_ty_tag = lhs_ty.zigTypeTag();
    const rhs_ty_tag = rhs_ty.zigTypeTag();
    const target = sema.mod.getTarget();

    const runtime_src: LazySrcLoc = src: {
        if (try sema.resolveMaybeUndefVal(block, lhs_src, lhs)) |lhs_val| {
            if (try sema.resolveMaybeUndefVal(block, rhs_src, rhs)) |rhs_val| {
                if (lhs_val.isUndef() or rhs_val.isUndef()) {
                    return sema.addConstUndef(Type.bool);
                }
                if (lhs_val.isNan() or rhs_val.isNan()) {
                    if (op == std.math.CompareOperator.neq) {
                        return Air.Inst.Ref.bool_true;
                    } else {
                        return Air.Inst.Ref.bool_false;
                    }
                }
                if (try Value.compareHeteroAdvanced(lhs_val, op, rhs_val, target, sema.kit(block, src))) {
                    return Air.Inst.Ref.bool_true;
                } else {
                    return Air.Inst.Ref.bool_false;
                }
            } else {
                break :src rhs_src;
            }
        } else {
            break :src lhs_src;
        }
    };

    // TODO handle comparisons against lazy zero values
    // Some values can be compared against zero without being runtime known or without forcing
    // a full resolution of their value, for example `@sizeOf(@Frame(function))` is known to
    // always be nonzero, and we benefit from not forcing the full evaluation and stack frame layout
    // of this function if we don't need to.
    try sema.requireRuntimeBlock(block, runtime_src);

    // For floats, emit a float comparison instruction.
    const lhs_is_float = switch (lhs_ty_tag) {
        .Float, .ComptimeFloat => true,
        else => false,
    };
    const rhs_is_float = switch (rhs_ty_tag) {
        .Float, .ComptimeFloat => true,
        else => false,
    };
    if (lhs_is_float and rhs_is_float) {
        // Implicit cast the smaller one to the larger one.
        const dest_ty = x: {
            if (lhs_ty_tag == .ComptimeFloat) {
                break :x rhs_ty;
            } else if (rhs_ty_tag == .ComptimeFloat) {
                break :x lhs_ty;
            }
            if (lhs_ty.floatBits(target) >= rhs_ty.floatBits(target)) {
                break :x lhs_ty;
            } else {
                break :x rhs_ty;
            }
        };
        const casted_lhs = try sema.coerce(block, dest_ty, lhs, lhs_src);
        const casted_rhs = try sema.coerce(block, dest_ty, rhs, rhs_src);
        return block.addBinOp(Air.Inst.Tag.fromCmpOp(op), casted_lhs, casted_rhs);
    }
    // For mixed unsigned integer sizes, implicit cast both operands to the larger integer.
    // For mixed signed and unsigned integers, implicit cast both operands to a signed
    // integer with + 1 bit.
    // For mixed floats and integers, extract the integer part from the float, cast that to
    // a signed integer with mantissa bits + 1, and if there was any non-integral part of the float,
    // add/subtract 1.
    const lhs_is_signed = if (try sema.resolveDefinedValue(block, lhs_src, lhs)) |lhs_val|
        lhs_val.compareWithZero(.lt)
    else
        (lhs_ty.isRuntimeFloat() or lhs_ty.isSignedInt());
    const rhs_is_signed = if (try sema.resolveDefinedValue(block, rhs_src, rhs)) |rhs_val|
        rhs_val.compareWithZero(.lt)
    else
        (rhs_ty.isRuntimeFloat() or rhs_ty.isSignedInt());
    const dest_int_is_signed = lhs_is_signed or rhs_is_signed;

    var dest_float_type: ?Type = null;

    var lhs_bits: usize = undefined;
    if (try sema.resolveMaybeUndefVal(block, lhs_src, lhs)) |lhs_val| {
        if (lhs_val.isUndef())
            return sema.addConstUndef(Type.bool);
        if (!rhs_is_signed) {
            switch (lhs_val.orderAgainstZero()) {
                .gt => {},
                .eq => switch (op) { // LHS = 0, RHS is unsigned
                    .lte => return Air.Inst.Ref.bool_true,
                    .gt => return Air.Inst.Ref.bool_false,
                    else => {},
                },
                .lt => switch (op) { // LHS < 0, RHS is unsigned
                    .neq, .lt, .lte => return Air.Inst.Ref.bool_true,
                    .eq, .gt, .gte => return Air.Inst.Ref.bool_false,
                },
            }
        }
        if (lhs_is_float) {
            var bigint_space: Value.BigIntSpace = undefined;
            var bigint = try lhs_val.toBigInt(&bigint_space, target).toManaged(sema.gpa);
            defer bigint.deinit();
            if (lhs_val.floatHasFraction()) {
                switch (op) {
                    .eq => return Air.Inst.Ref.bool_false,
                    .neq => return Air.Inst.Ref.bool_true,
                    else => {},
                }
                if (lhs_is_signed) {
                    try bigint.addScalar(bigint.toConst(), -1);
                } else {
                    try bigint.addScalar(bigint.toConst(), 1);
                }
            }
            lhs_bits = bigint.toConst().bitCountTwosComp();
        } else {
            lhs_bits = lhs_val.intBitCountTwosComp(target);
        }
        lhs_bits += @boolToInt(!lhs_is_signed and dest_int_is_signed);
    } else if (lhs_is_float) {
        dest_float_type = lhs_ty;
    } else {
        const int_info = lhs_ty.intInfo(target);
        lhs_bits = int_info.bits + @boolToInt(int_info.signedness == .unsigned and dest_int_is_signed);
    }

    var rhs_bits: usize = undefined;
    if (try sema.resolveMaybeUndefVal(block, rhs_src, rhs)) |rhs_val| {
        if (rhs_val.isUndef())
            return sema.addConstUndef(Type.bool);
        if (!lhs_is_signed) {
            switch (rhs_val.orderAgainstZero()) {
                .gt => {},
                .eq => switch (op) { // RHS = 0, LHS is unsigned
                    .gte => return Air.Inst.Ref.bool_true,
                    .lt => return Air.Inst.Ref.bool_false,
                    else => {},
                },
                .lt => switch (op) { // RHS < 0, LHS is unsigned
                    .neq, .gt, .gte => return Air.Inst.Ref.bool_true,
                    .eq, .lt, .lte => return Air.Inst.Ref.bool_false,
                },
            }
        }
        if (rhs_is_float) {
            var bigint_space: Value.BigIntSpace = undefined;
            var bigint = try rhs_val.toBigInt(&bigint_space, target).toManaged(sema.gpa);
            defer bigint.deinit();
            if (rhs_val.floatHasFraction()) {
                switch (op) {
                    .eq => return Air.Inst.Ref.bool_false,
                    .neq => return Air.Inst.Ref.bool_true,
                    else => {},
                }
                if (rhs_is_signed) {
                    try bigint.addScalar(bigint.toConst(), -1);
                } else {
                    try bigint.addScalar(bigint.toConst(), 1);
                }
            }
            rhs_bits = bigint.toConst().bitCountTwosComp();
        } else {
            rhs_bits = rhs_val.intBitCountTwosComp(target);
        }
        rhs_bits += @boolToInt(!rhs_is_signed and dest_int_is_signed);
    } else if (rhs_is_float) {
        dest_float_type = rhs_ty;
    } else {
        const int_info = rhs_ty.intInfo(target);
        rhs_bits = int_info.bits + @boolToInt(int_info.signedness == .unsigned and dest_int_is_signed);
    }

    const dest_ty = if (dest_float_type) |ft| ft else blk: {
        const max_bits = std.math.max(lhs_bits, rhs_bits);
        const casted_bits = std.math.cast(u16, max_bits) catch |err| switch (err) {
            error.Overflow => return sema.fail(block, src, "{d} exceeds maximum integer bit count", .{max_bits}),
        };
        const signedness: std.builtin.Signedness = if (dest_int_is_signed) .signed else .unsigned;
        break :blk try Module.makeIntType(sema.arena, signedness, casted_bits);
    };
    const casted_lhs = try sema.coerce(block, dest_ty, lhs, lhs_src);
    const casted_rhs = try sema.coerce(block, dest_ty, rhs, rhs_src);

    return block.addBinOp(Air.Inst.Tag.fromCmpOp(op), casted_lhs, casted_rhs);
}

/// Asserts that lhs and rhs types are both vectors.
fn cmpVector(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    lhs: Air.Inst.Ref,
    rhs: Air.Inst.Ref,
    op: std.math.CompareOperator,
    lhs_src: LazySrcLoc,
    rhs_src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    const lhs_ty = sema.typeOf(lhs);
    const rhs_ty = sema.typeOf(rhs);
    assert(lhs_ty.zigTypeTag() == .Vector);
    assert(rhs_ty.zigTypeTag() == .Vector);
    try sema.checkVectorizableBinaryOperands(block, src, lhs_ty, rhs_ty, lhs_src, rhs_src);

    const result_ty = try Type.vector(sema.arena, lhs_ty.vectorLen(), Type.@"bool");
    const target = sema.mod.getTarget();

    const runtime_src: LazySrcLoc = src: {
        if (try sema.resolveMaybeUndefVal(block, lhs_src, lhs)) |lhs_val| {
            if (try sema.resolveMaybeUndefVal(block, rhs_src, rhs)) |rhs_val| {
                if (lhs_val.isUndef() or rhs_val.isUndef()) {
                    return sema.addConstUndef(result_ty);
                }
                const cmp_val = try lhs_val.compareVector(op, rhs_val, lhs_ty, sema.arena, target);
                return sema.addConstant(result_ty, cmp_val);
            } else {
                break :src rhs_src;
            }
        } else {
            break :src lhs_src;
        }
    };

    try sema.requireRuntimeBlock(block, runtime_src);
    const result_ty_inst = try sema.addType(result_ty);
    return block.addCmpVector(lhs, rhs, op, result_ty_inst);
}

fn wrapOptional(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |val| {
        return sema.addConstant(dest_ty, try Value.Tag.opt_payload.create(sema.arena, val));
    }

    try sema.requireRuntimeBlock(block, inst_src);
    return block.addTyOp(.wrap_optional, dest_ty, inst);
}

fn wrapErrorUnionPayload(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const dest_payload_ty = dest_ty.errorUnionPayload();
    const coerced = try sema.coerce(block, dest_payload_ty, inst, inst_src);
    if (try sema.resolveMaybeUndefVal(block, inst_src, coerced)) |val| {
        return sema.addConstant(dest_ty, try Value.Tag.eu_payload.create(sema.arena, val));
    }
    try sema.requireRuntimeBlock(block, inst_src);
    try sema.queueFullTypeResolution(dest_payload_ty);
    return block.addTyOp(.wrap_errunion_payload, dest_ty, coerced);
}

fn wrapErrorUnionSet(
    sema: *Sema,
    block: *Block,
    dest_ty: Type,
    inst: Air.Inst.Ref,
    inst_src: LazySrcLoc,
) !Air.Inst.Ref {
    const inst_ty = sema.typeOf(inst);
    const dest_err_set_ty = dest_ty.errorUnionSet();
    if (try sema.resolveMaybeUndefVal(block, inst_src, inst)) |val| {
        switch (dest_err_set_ty.tag()) {
            .anyerror => {},
            .error_set_single => ok: {
                const expected_name = val.castTag(.@"error").?.data.name;
                const n = dest_err_set_ty.castTag(.error_set_single).?.data;
                if (mem.eql(u8, expected_name, n)) break :ok;
                return sema.failWithErrorSetCodeMissing(block, inst_src, dest_err_set_ty, inst_ty);
            },
            .error_set => {
                const expected_name = val.castTag(.@"error").?.data.name;
                const error_set = dest_err_set_ty.castTag(.error_set).?.data;
                if (!error_set.names.contains(expected_name)) {
                    return sema.failWithErrorSetCodeMissing(block, inst_src, dest_err_set_ty, inst_ty);
                }
            },
            .error_set_inferred => ok: {
                const expected_name = val.castTag(.@"error").?.data.name;
                const ies = dest_err_set_ty.castTag(.error_set_inferred).?.data;

                // We carefully do this in an order that avoids unnecessarily
                // resolving the destination error set type.
                if (ies.is_anyerror) break :ok;
                if (ies.errors.contains(expected_name)) break :ok;
                if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, dest_err_set_ty, inst_ty, inst_src, inst_src)) {
                    break :ok;
                }

                return sema.failWithErrorSetCodeMissing(block, inst_src, dest_err_set_ty, inst_ty);
            },
            .error_set_merged => {
                const expected_name = val.castTag(.@"error").?.data.name;
                const error_set = dest_err_set_ty.castTag(.error_set_merged).?.data;
                if (!error_set.contains(expected_name)) {
                    return sema.failWithErrorSetCodeMissing(block, inst_src, dest_err_set_ty, inst_ty);
                }
            },
            else => unreachable,
        }
        return sema.addConstant(dest_ty, val);
    }

    try sema.requireRuntimeBlock(block, inst_src);
    const coerced = try sema.coerce(block, dest_err_set_ty, inst, inst_src);
    return block.addTyOp(.wrap_errunion_err, dest_ty, coerced);
}

fn unionToTag(
    sema: *Sema,
    block: *Block,
    enum_ty: Type,
    un: Air.Inst.Ref,
    un_src: LazySrcLoc,
) !Air.Inst.Ref {
    if ((try sema.typeHasOnePossibleValue(block, un_src, enum_ty))) |opv| {
        return sema.addConstant(enum_ty, opv);
    }
    if (try sema.resolveMaybeUndefVal(block, un_src, un)) |un_val| {
        return sema.addConstant(enum_ty, un_val.unionTag());
    }
    try sema.requireRuntimeBlock(block, un_src);
    return block.addTyOp(.get_union_tag, enum_ty, un);
}

fn resolvePeerTypes(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    instructions: []Air.Inst.Ref,
    candidate_srcs: Module.PeerTypeCandidateSrc,
) !Type {
    switch (instructions.len) {
        0 => return Type.initTag(.noreturn),
        1 => return sema.typeOf(instructions[0]),
        else => {},
    }

    const target = sema.mod.getTarget();

    var chosen = instructions[0];
    // If this is non-null then it does the following thing, depending on the chosen zigTypeTag().
    //  * ErrorSet: this is an override
    //  * ErrorUnion: this is an override of the error set only
    //  * other: at the end we make an ErrorUnion with the other thing and this
    var err_set_ty: ?Type = null;
    var any_are_null = false;
    var seen_const = false;
    var convert_to_slice = false;
    var chosen_i: usize = 0;
    for (instructions[1..]) |candidate, candidate_i| {
        const candidate_ty = sema.typeOf(candidate);
        const chosen_ty = sema.typeOf(chosen);

        const candidate_ty_tag = try candidate_ty.zigTypeTagOrPoison();
        const chosen_ty_tag = try chosen_ty.zigTypeTagOrPoison();

        if (candidate_ty.eql(chosen_ty, target))
            continue;

        switch (candidate_ty_tag) {
            .NoReturn, .Undefined => continue,

            .Null => {
                any_are_null = true;
                continue;
            },

            .Int => switch (chosen_ty_tag) {
                .ComptimeInt => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                .Int => {
                    const chosen_info = chosen_ty.intInfo(target);
                    const candidate_info = candidate_ty.intInfo(target);

                    if (chosen_info.bits < candidate_info.bits) {
                        chosen = candidate;
                        chosen_i = candidate_i + 1;
                    }
                    continue;
                },
                .Pointer => if (chosen_ty.ptrSize() == .C) continue,
                else => {},
            },
            .ComptimeInt => switch (chosen_ty_tag) {
                .Int, .Float, .ComptimeFloat => continue,
                .Pointer => if (chosen_ty.ptrSize() == .C) continue,
                else => {},
            },
            .Float => switch (chosen_ty_tag) {
                .Float => {
                    if (chosen_ty.floatBits(target) < candidate_ty.floatBits(target)) {
                        chosen = candidate;
                        chosen_i = candidate_i + 1;
                    }
                    continue;
                },
                .ComptimeFloat, .ComptimeInt => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                else => {},
            },
            .ComptimeFloat => switch (chosen_ty_tag) {
                .Float => continue,
                .ComptimeInt => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                else => {},
            },
            .Enum => switch (chosen_ty_tag) {
                .EnumLiteral => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                .Union => continue,
                else => {},
            },
            .EnumLiteral => switch (chosen_ty_tag) {
                .Enum, .Union => continue,
                else => {},
            },
            .Union => switch (chosen_ty_tag) {
                .Enum, .EnumLiteral => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                else => {},
            },
            .ErrorSet => switch (chosen_ty_tag) {
                .ErrorSet => {
                    // If chosen is superset of candidate, keep it.
                    // If candidate is superset of chosen, switch it.
                    // If neither is a superset, merge errors.
                    const chosen_set_ty = err_set_ty orelse chosen_ty;

                    if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_ty, src, src)) {
                        continue;
                    }
                    if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_ty, chosen_set_ty, src, src)) {
                        err_set_ty = null;
                        chosen = candidate;
                        chosen_i = candidate_i + 1;
                        continue;
                    }

                    err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_ty);
                    continue;
                },
                .ErrorUnion => {
                    const chosen_set_ty = err_set_ty orelse chosen_ty.errorUnionSet();

                    if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_ty, src, src)) {
                        continue;
                    }
                    if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_ty, chosen_set_ty, src, src)) {
                        err_set_ty = candidate_ty;
                        continue;
                    }

                    err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_ty);
                    continue;
                },
                else => {
                    if (err_set_ty) |chosen_set_ty| {
                        if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_ty, src, src)) {
                            continue;
                        }
                        if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_ty, chosen_set_ty, src, src)) {
                            err_set_ty = candidate_ty;
                            continue;
                        }

                        err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_ty);
                        continue;
                    } else {
                        err_set_ty = candidate_ty;
                        continue;
                    }
                },
            },
            .ErrorUnion => switch (chosen_ty_tag) {
                .ErrorSet => {
                    const chosen_set_ty = err_set_ty orelse chosen_ty;
                    const candidate_set_ty = candidate_ty.errorUnionSet();

                    if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_set_ty, src, src)) {
                        err_set_ty = chosen_set_ty;
                    } else if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_set_ty, chosen_set_ty, src, src)) {
                        err_set_ty = null;
                    } else {
                        err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_set_ty);
                    }
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },

                .ErrorUnion => {
                    const chosen_payload_ty = chosen_ty.errorUnionPayload();
                    const candidate_payload_ty = candidate_ty.errorUnionPayload();

                    const coerce_chosen = (try sema.coerceInMemoryAllowed(block, chosen_payload_ty, candidate_payload_ty, false, target, src, src)) == .ok;
                    const coerce_candidate = (try sema.coerceInMemoryAllowed(block, candidate_payload_ty, chosen_payload_ty, false, target, src, src)) == .ok;

                    if (coerce_chosen or coerce_candidate) {
                        // If we can coerce to the candidate, we switch to that
                        // type. This is the same logic as the bare (non-union)
                        // coercion check we do at the top of this func.
                        if (coerce_candidate) {
                            chosen = candidate;
                            chosen_i = candidate_i + 1;
                        }

                        const chosen_set_ty = err_set_ty orelse chosen_ty.errorUnionSet();
                        const candidate_set_ty = candidate_ty.errorUnionSet();

                        if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_set_ty, src, src)) {
                            err_set_ty = chosen_set_ty;
                        } else if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_set_ty, chosen_set_ty, src, src)) {
                            err_set_ty = candidate_set_ty;
                        } else {
                            err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_set_ty);
                        }
                        continue;
                    }
                },

                else => {
                    if (err_set_ty) |chosen_set_ty| {
                        const candidate_set_ty = candidate_ty.errorUnionSet();
                        if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, chosen_set_ty, candidate_set_ty, src, src)) {
                            err_set_ty = chosen_set_ty;
                        } else if (.ok == try sema.coerceInMemoryAllowedErrorSets(block, candidate_set_ty, chosen_set_ty, src, src)) {
                            err_set_ty = null;
                        } else {
                            err_set_ty = try chosen_set_ty.errorSetMerge(sema.arena, candidate_set_ty);
                        }
                    }
                    seen_const = seen_const or chosen_ty.isConstPtr();
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
            },
            .Pointer => {
                const cand_info = candidate_ty.ptrInfo().data;
                switch (chosen_ty_tag) {
                    .Pointer => {
                        const chosen_info = chosen_ty.ptrInfo().data;

                        seen_const = seen_const or !chosen_info.mutable or !cand_info.mutable;

                        // *[N]T to [*]T
                        // *[N]T to []T
                        if ((cand_info.size == .Many or cand_info.size == .Slice) and
                            chosen_info.size == .One and
                            chosen_info.pointee_type.zigTypeTag() == .Array)
                        {
                            // In case we see i.e.: `*[1]T`, `*[2]T`, `[*]T`
                            convert_to_slice = false;
                            chosen = candidate;
                            chosen_i = candidate_i + 1;
                            continue;
                        }
                        if (cand_info.size == .One and
                            cand_info.pointee_type.zigTypeTag() == .Array and
                            (chosen_info.size == .Many or chosen_info.size == .Slice))
                        {
                            // In case we see i.e.: `*[1]T`, `*[2]T`, `[*]T`
                            convert_to_slice = false;
                            continue;
                        }

                        // *[N]T and *[M]T
                        // Verify both are single-pointers to arrays.
                        // Keep the one whose element type can be coerced into.
                        if (chosen_info.size == .One and
                            cand_info.size == .One and
                            chosen_info.pointee_type.zigTypeTag() == .Array and
                            cand_info.pointee_type.zigTypeTag() == .Array)
                        {
                            const chosen_elem_ty = chosen_info.pointee_type.childType();
                            const cand_elem_ty = cand_info.pointee_type.childType();

                            const chosen_ok = .ok == try sema.coerceInMemoryAllowed(block, chosen_elem_ty, cand_elem_ty, chosen_info.mutable, target, src, src);
                            if (chosen_ok) {
                                convert_to_slice = true;
                                continue;
                            }

                            const cand_ok = .ok == try sema.coerceInMemoryAllowed(block, cand_elem_ty, chosen_elem_ty, cand_info.mutable, target, src, src);
                            if (cand_ok) {
                                convert_to_slice = true;
                                chosen = candidate;
                                chosen_i = candidate_i + 1;
                                continue;
                            }

                            // They're both bad. Report error.
                            // In the future we probably want to use the
                            // coerceInMemoryAllowed error reporting mechanism,
                            // however, for now we just fall through for the
                            // "incompatible types" error below.
                        }

                        // [*c]T and any other pointer size
                        // Whichever element type can coerce to the other one, is
                        // the one we will keep. If they're both OK then we keep the
                        // C pointer since it matches both single and many pointers.
                        if (cand_info.size == .C or chosen_info.size == .C) {
                            const cand_ok = .ok == try sema.coerceInMemoryAllowed(block, cand_info.pointee_type, chosen_info.pointee_type, cand_info.mutable, target, src, src);
                            const chosen_ok = .ok == try sema.coerceInMemoryAllowed(block, chosen_info.pointee_type, cand_info.pointee_type, chosen_info.mutable, target, src, src);

                            if (cand_ok) {
                                if (chosen_ok) {
                                    if (chosen_info.size == .C) {
                                        continue;
                                    } else {
                                        chosen = candidate;
                                        chosen_i = candidate_i + 1;
                                        continue;
                                    }
                                } else {
                                    chosen = candidate;
                                    chosen_i = candidate_i + 1;
                                    continue;
                                }
                            } else {
                                if (chosen_ok) {
                                    continue;
                                } else {
                                    // They're both bad. Report error.
                                    // In the future we probably want to use the
                                    // coerceInMemoryAllowed error reporting mechanism,
                                    // however, for now we just fall through for the
                                    // "incompatible types" error below.
                                }
                            }
                        }
                    },
                    .Int, .ComptimeInt => {
                        if (cand_info.size == .C) {
                            chosen = candidate;
                            chosen_i = candidate_i + 1;
                            continue;
                        }
                    },
                    .Optional => {
                        var opt_child_buf: Type.Payload.ElemType = undefined;
                        const chosen_ptr_ty = chosen_ty.optionalChild(&opt_child_buf);
                        if (chosen_ptr_ty.zigTypeTag() == .Pointer) {
                            const chosen_info = chosen_ptr_ty.ptrInfo().data;

                            seen_const = seen_const or !chosen_info.mutable or !cand_info.mutable;

                            // *[N]T to ?![*]T
                            // *[N]T to ?![]T
                            if (cand_info.size == .One and
                                cand_info.pointee_type.zigTypeTag() == .Array and
                                (chosen_info.size == .Many or chosen_info.size == .Slice))
                            {
                                continue;
                            }
                        }
                    },
                    .ErrorUnion => {
                        const chosen_ptr_ty = chosen_ty.errorUnionPayload();
                        if (chosen_ptr_ty.zigTypeTag() == .Pointer) {
                            const chosen_info = chosen_ptr_ty.ptrInfo().data;

                            seen_const = seen_const or !chosen_info.mutable or !cand_info.mutable;

                            // *[N]T to E![*]T
                            // *[N]T to E![]T
                            if (cand_info.size == .One and
                                cand_info.pointee_type.zigTypeTag() == .Array and
                                (chosen_info.size == .Many or chosen_info.size == .Slice))
                            {
                                continue;
                            }
                        }
                    },
                    else => {},
                }
            },
            .Optional => {
                var opt_child_buf: Type.Payload.ElemType = undefined;
                const opt_child_ty = candidate_ty.optionalChild(&opt_child_buf);
                if ((try sema.coerceInMemoryAllowed(block, chosen_ty, opt_child_ty, false, target, src, src)) == .ok) {
                    seen_const = seen_const or opt_child_ty.isConstPtr();
                    any_are_null = true;
                    continue;
                }

                seen_const = seen_const or chosen_ty.isConstPtr();
                any_are_null = false;
                chosen = candidate;
                chosen_i = candidate_i + 1;
                continue;
            },
            .Vector => switch (chosen_ty_tag) {
                .Array => {
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                },
                else => {},
            },
            .Array => switch (chosen_ty_tag) {
                .Vector => continue,
                else => {},
            },
            else => {},
        }

        switch (chosen_ty_tag) {
            .NoReturn, .Undefined => {
                chosen = candidate;
                chosen_i = candidate_i + 1;
                continue;
            },
            .Null => {
                any_are_null = true;
                chosen = candidate;
                chosen_i = candidate_i + 1;
                continue;
            },
            .Optional => {
                var opt_child_buf: Type.Payload.ElemType = undefined;
                const opt_child_ty = chosen_ty.optionalChild(&opt_child_buf);
                if ((try sema.coerceInMemoryAllowed(block, opt_child_ty, candidate_ty, false, target, src, src)) == .ok) {
                    continue;
                }
                if ((try sema.coerceInMemoryAllowed(block, candidate_ty, opt_child_ty, false, target, src, src)) == .ok) {
                    any_are_null = true;
                    chosen = candidate;
                    chosen_i = candidate_i + 1;
                    continue;
                }
            },
            .ErrorUnion => {
                const payload_ty = chosen_ty.errorUnionPayload();
                if ((try sema.coerceInMemoryAllowed(block, payload_ty, candidate_ty, false, target, src, src)) == .ok) {
                    continue;
                }
            },
            else => {},
        }

        // If the candidate can coerce into our chosen type, we're done.
        // If the chosen type can coerce into the candidate, use that.
        if ((try sema.coerceInMemoryAllowed(block, chosen_ty, candidate_ty, false, target, src, src)) == .ok) {
            continue;
        }
        if ((try sema.coerceInMemoryAllowed(block, candidate_ty, chosen_ty, false, target, src, src)) == .ok) {
            chosen = candidate;
            chosen_i = candidate_i + 1;
            continue;
        }

        // At this point, we hit a compile error. We need to recover
        // the source locations.
        const chosen_src = candidate_srcs.resolve(
            sema.gpa,
            block.src_decl,
            chosen_i,
        );
        const candidate_src = candidate_srcs.resolve(
            sema.gpa,
            block.src_decl,
            candidate_i + 1,
        );

        const msg = msg: {
            const msg = try sema.errMsg(block, src, "incompatible types: '{}' and '{}'", .{
                chosen_ty.fmt(target),
                candidate_ty.fmt(target),
            });
            errdefer msg.destroy(sema.gpa);

            if (chosen_src) |src_loc|
                try sema.errNote(block, src_loc, msg, "type '{}' here", .{chosen_ty.fmt(target)});

            if (candidate_src) |src_loc|
                try sema.errNote(block, src_loc, msg, "type '{}' here", .{candidate_ty.fmt(target)});

            break :msg msg;
        };
        return sema.failWithOwnedErrorMsg(block, msg);
    }

    const chosen_ty = sema.typeOf(chosen);

    if (convert_to_slice) {
        // turn *[N]T => []T
        const chosen_child_ty = chosen_ty.childType();
        var info = chosen_ty.ptrInfo();
        info.data.sentinel = chosen_child_ty.sentinel();
        info.data.size = .Slice;
        info.data.mutable = !(seen_const or chosen_child_ty.isConstPtr());
        info.data.pointee_type = switch (chosen_child_ty.tag()) {
            .array => chosen_child_ty.elemType2(),
            .array_u8, .array_u8_sentinel_0 => Type.initTag(.u8),
            else => unreachable,
        };

        const new_ptr_ty = try Type.ptr(sema.arena, target, info.data);
        const opt_ptr_ty = if (any_are_null)
            try Type.optional(sema.arena, new_ptr_ty)
        else
            new_ptr_ty;
        const set_ty = err_set_ty orelse return opt_ptr_ty;
        return try Type.errorUnion(sema.arena, set_ty, opt_ptr_ty, target);
    }

    if (seen_const) {
        // turn []T => []const T
        switch (chosen_ty.zigTypeTag()) {
            .ErrorUnion => {
                const ptr_ty = chosen_ty.errorUnionPayload();
                var info = ptr_ty.ptrInfo();
                info.data.mutable = false;
                const new_ptr_ty = try Type.ptr(sema.arena, target, info.data);
                const opt_ptr_ty = if (any_are_null)
                    try Type.optional(sema.arena, new_ptr_ty)
                else
                    new_ptr_ty;
                const set_ty = err_set_ty orelse chosen_ty.errorUnionSet();
                return try Type.errorUnion(sema.arena, set_ty, opt_ptr_ty, target);
            },
            .Pointer => {
                var info = chosen_ty.ptrInfo();
                info.data.mutable = false;
                const new_ptr_ty = try Type.ptr(sema.arena, target, info.data);
                const opt_ptr_ty = if (any_are_null)
                    try Type.optional(sema.arena, new_ptr_ty)
                else
                    new_ptr_ty;
                const set_ty = err_set_ty orelse return opt_ptr_ty;
                return try Type.errorUnion(sema.arena, set_ty, opt_ptr_ty, target);
            },
            else => return chosen_ty,
        }
    }

    if (any_are_null) {
        const opt_ty = switch (chosen_ty.zigTypeTag()) {
            .Null, .Optional => chosen_ty,
            else => try Type.optional(sema.arena, chosen_ty),
        };
        const set_ty = err_set_ty orelse return opt_ty;
        return try Type.errorUnion(sema.arena, set_ty, opt_ty, target);
    }

    if (err_set_ty) |ty| switch (chosen_ty.zigTypeTag()) {
        .ErrorSet => return ty,
        .ErrorUnion => {
            const payload_ty = chosen_ty.errorUnionPayload();
            return try Type.errorUnion(sema.arena, ty, payload_ty, target);
        },
        else => return try Type.errorUnion(sema.arena, ty, chosen_ty, target),
    };

    return chosen_ty;
}

pub fn resolveFnTypes(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    fn_info: Type.Payload.Function.Data,
) CompileError!void {
    try sema.resolveTypeFully(block, src, fn_info.return_type);

    for (fn_info.param_types) |param_ty| {
        try sema.resolveTypeFully(block, src, param_ty);
    }
}

pub fn resolveTypeLayout(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    if (build_options.omit_stage2)
        @panic("sadly stage2 is omitted from this build to save memory on the CI server");

    switch (ty.zigTypeTag()) {
        .Struct => return sema.resolveStructLayout(block, src, ty),
        .Union => return sema.resolveUnionLayout(block, src, ty),
        .Array => {
            if (ty.arrayLenIncludingSentinel() == 0) return;
            const elem_ty = ty.childType();
            return sema.resolveTypeLayout(block, src, elem_ty);
        },
        .Optional => {
            var buf: Type.Payload.ElemType = undefined;
            const payload_ty = ty.optionalChild(&buf);
            // In case of querying the ABI alignment of this optional, we will ask
            // for hasRuntimeBits() of the payload type, so we need "requires comptime"
            // to be known already before this function returns.
            _ = try sema.typeRequiresComptime(block, src, payload_ty);
            return sema.resolveTypeLayout(block, src, payload_ty);
        },
        .ErrorUnion => {
            const payload_ty = ty.errorUnionPayload();
            return sema.resolveTypeLayout(block, src, payload_ty);
        },
        else => {},
    }
}

fn resolveStructLayout(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const resolved_ty = try sema.resolveTypeFields(block, src, ty);
    if (resolved_ty.castTag(.@"struct")) |payload| {
        const target = sema.mod.getTarget();
        const struct_obj = payload.data;
        switch (struct_obj.status) {
            .none, .have_field_types => {},
            .field_types_wip, .layout_wip => {
                return sema.fail(block, src, "struct {} depends on itself", .{ty.fmt(target)});
            },
            .have_layout, .fully_resolved_wip, .fully_resolved => return,
        }
        struct_obj.status = .layout_wip;
        for (struct_obj.fields.values()) |field| {
            try sema.resolveTypeLayout(block, src, field.ty);
        }
        struct_obj.status = .have_layout;

        // In case of querying the ABI alignment of this struct, we will ask
        // for hasRuntimeBits() of each field, so we need "requires comptime"
        // to be known already before this function returns.
        for (struct_obj.fields.values()) |field| {
            _ = try sema.typeRequiresComptime(block, src, field.ty);
        }
    }
    // otherwise it's a tuple; no need to resolve anything
}

fn resolveUnionLayout(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    const resolved_ty = try sema.resolveTypeFields(block, src, ty);
    const union_obj = resolved_ty.cast(Type.Payload.Union).?.data;
    const target = sema.mod.getTarget();
    switch (union_obj.status) {
        .none, .have_field_types => {},
        .field_types_wip, .layout_wip => {
            return sema.fail(block, src, "union {} depends on itself", .{ty.fmt(target)});
        },
        .have_layout, .fully_resolved_wip, .fully_resolved => return,
    }
    union_obj.status = .layout_wip;
    for (union_obj.fields.values()) |field| {
        try sema.resolveTypeLayout(block, src, field.ty);
    }
    union_obj.status = .have_layout;
}

pub fn resolveTypeFully(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    switch (ty.zigTypeTag()) {
        .Pointer => {
            const child_ty = try sema.resolveTypeFields(block, src, ty.childType());
            return resolveTypeFully(sema, block, src, child_ty);
        },
        .Struct => return resolveStructFully(sema, block, src, ty),
        .Union => return resolveUnionFully(sema, block, src, ty),
        .Array => return resolveTypeFully(sema, block, src, ty.childType()),
        .Optional => {
            var buf: Type.Payload.ElemType = undefined;
            return resolveTypeFully(sema, block, src, ty.optionalChild(&buf));
        },
        .ErrorUnion => return resolveTypeFully(sema, block, src, ty.errorUnionPayload()),
        .Fn => {
            const info = ty.fnInfo();
            if (info.is_generic) {
                // Resolving of generic function types is defeerred to when
                // the function is instantiated.
                return;
            }
            for (info.param_types) |param_ty| {
                const param_ty_src = src; // TODO better source location
                try sema.resolveTypeFully(block, param_ty_src, param_ty);
            }
            const return_ty_src = src; // TODO better source location
            try sema.resolveTypeFully(block, return_ty_src, info.return_type);
        },
        else => {},
    }
}

fn resolveStructFully(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    try resolveStructLayout(sema, block, src, ty);

    const resolved_ty = try sema.resolveTypeFields(block, src, ty);
    const payload = resolved_ty.castTag(.@"struct") orelse return;
    const struct_obj = payload.data;
    switch (struct_obj.status) {
        .none, .have_field_types, .field_types_wip, .layout_wip, .have_layout => {},
        .fully_resolved_wip, .fully_resolved => return,
    }

    // After we have resolve struct layout we have to go over the fields again to
    // make sure pointer fields get their child types resolved as well
    struct_obj.status = .fully_resolved_wip;
    for (struct_obj.fields.values()) |field| {
        try sema.resolveTypeFully(block, src, field.ty);
    }
    struct_obj.status = .fully_resolved;

    // And let's not forget comptime-only status.
    _ = try sema.typeRequiresComptime(block, src, ty);
}

fn resolveUnionFully(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    try resolveUnionLayout(sema, block, src, ty);

    const resolved_ty = try sema.resolveTypeFields(block, src, ty);
    const union_obj = resolved_ty.cast(Type.Payload.Union).?.data;
    switch (union_obj.status) {
        .none, .have_field_types, .field_types_wip, .layout_wip, .have_layout => {},
        .fully_resolved_wip, .fully_resolved => return,
    }

    // Same goes for unions (see comment about structs)
    union_obj.status = .fully_resolved_wip;
    for (union_obj.fields.values()) |field| {
        try sema.resolveTypeFully(block, src, field.ty);
    }
    union_obj.status = .fully_resolved;

    // And let's not forget comptime-only status.
    _ = try sema.typeRequiresComptime(block, src, ty);
}

pub fn resolveTypeFields(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!Type {
    if (build_options.omit_stage2)
        @panic("sadly stage2 is omitted from this build to save memory on the CI server");
    switch (ty.tag()) {
        .@"struct" => {
            const struct_obj = ty.castTag(.@"struct").?.data;
            try sema.resolveTypeFieldsStruct(block, src, ty, struct_obj);
            return ty;
        },
        .@"union", .union_tagged => {
            const union_obj = ty.cast(Type.Payload.Union).?.data;
            try sema.resolveTypeFieldsUnion(block, src, ty, union_obj);
            return ty;
        },
        .type_info => return sema.resolveBuiltinTypeFields(block, src, "Type"),
        .extern_options => return sema.resolveBuiltinTypeFields(block, src, "ExternOptions"),
        .export_options => return sema.resolveBuiltinTypeFields(block, src, "ExportOptions"),
        .atomic_order => return sema.resolveBuiltinTypeFields(block, src, "AtomicOrder"),
        .atomic_rmw_op => return sema.resolveBuiltinTypeFields(block, src, "AtomicRmwOp"),
        .calling_convention => return sema.resolveBuiltinTypeFields(block, src, "CallingConvention"),
        .address_space => return sema.resolveBuiltinTypeFields(block, src, "AddressSpace"),
        .float_mode => return sema.resolveBuiltinTypeFields(block, src, "FloatMode"),
        .reduce_op => return sema.resolveBuiltinTypeFields(block, src, "ReduceOp"),
        .call_options => return sema.resolveBuiltinTypeFields(block, src, "CallOptions"),
        .prefetch_options => return sema.resolveBuiltinTypeFields(block, src, "PrefetchOptions"),

        else => return ty,
    }
}

fn resolveTypeFieldsStruct(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
    struct_obj: *Module.Struct,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (struct_obj.status) {
        .none => {},
        .field_types_wip => {
            return sema.fail(block, src, "struct {} depends on itself", .{ty.fmt(target)});
        },
        .have_field_types,
        .have_layout,
        .layout_wip,
        .fully_resolved_wip,
        .fully_resolved,
        => return,
    }

    struct_obj.status = .field_types_wip;
    try semaStructFields(sema.mod, struct_obj);

    if (struct_obj.fields.count() == 0) {
        struct_obj.status = .have_layout;
    } else {
        struct_obj.status = .have_field_types;
    }
}

fn resolveTypeFieldsUnion(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
    union_obj: *Module.Union,
) CompileError!void {
    const target = sema.mod.getTarget();
    switch (union_obj.status) {
        .none => {},
        .field_types_wip => {
            return sema.fail(block, src, "union {} depends on itself", .{ty.fmt(target)});
        },
        .have_field_types,
        .have_layout,
        .layout_wip,
        .fully_resolved_wip,
        .fully_resolved,
        => return,
    }

    union_obj.status = .field_types_wip;
    try semaUnionFields(sema.mod, union_obj);
    union_obj.status = .have_field_types;
}

fn resolveBuiltinTypeFields(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    name: []const u8,
) CompileError!Type {
    const resolved_ty = try sema.getBuiltinType(block, src, name);
    return sema.resolveTypeFields(block, src, resolved_ty);
}

fn resolveInferredErrorSet(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ies: *Module.Fn.InferredErrorSet,
) CompileError!void {
    if (ies.is_resolved) return;

    if (ies.func.state == .in_progress) {
        return sema.fail(block, src, "unable to resolve inferred error set", .{});
    }

    // In order to ensure that all dependencies are properly added to the set, we
    // need to ensure the function body is analyzed of the inferred error set.
    // However, in the case of comptime/inline function calls with inferred error sets,
    // each call gets a new InferredErrorSet object, which points to the same
    // `*Module.Fn`. Not only is the function not relevant to the inferred error set
    // in this case, it may be a generic function which would cause an assertion failure
    // if we called `ensureFuncBodyAnalyzed` on it here.
    if (ies.func.owner_decl.ty.fnInfo().return_type.errorUnionSet().castTag(.error_set_inferred).?.data == ies) {
        // In this case we are dealing with the actual InferredErrorSet object that
        // corresponds to the function, not one created to track an inline/comptime call.
        try sema.ensureFuncBodyAnalyzed(ies.func);
    }

    ies.is_resolved = true;

    var it = ies.inferred_error_sets.keyIterator();
    while (it.next()) |other_error_set_ptr| {
        const other_ies: *Module.Fn.InferredErrorSet = other_error_set_ptr.*;
        if (ies == other_ies) continue;
        try sema.resolveInferredErrorSet(block, src, other_ies);

        for (other_ies.errors.keys()) |key| {
            try ies.errors.put(sema.gpa, key, {});
        }
        if (other_ies.is_anyerror)
            ies.is_anyerror = true;
    }
}

fn resolveInferredErrorSetTy(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!void {
    if (ty.castTag(.error_set_inferred)) |inferred| {
        try sema.resolveInferredErrorSet(block, src, inferred.data);
    }
}

fn semaStructFields(
    mod: *Module,
    struct_obj: *Module.Struct,
) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const gpa = mod.gpa;
    const decl = struct_obj.owner_decl;
    const zir = struct_obj.namespace.file_scope.zir;
    const extended = zir.instructions.items(.data)[struct_obj.zir_index].extended;
    assert(extended.opcode == .struct_decl);
    const small = @bitCast(Zir.Inst.StructDecl.Small, extended.small);
    var extra_index: usize = extended.operand;

    const src: LazySrcLoc = .{ .node_offset = struct_obj.node_offset };
    extra_index += @boolToInt(small.has_src_node);

    const body_len = if (small.has_body_len) blk: {
        const body_len = zir.extra[extra_index];
        extra_index += 1;
        break :blk body_len;
    } else 0;

    const fields_len = if (small.has_fields_len) blk: {
        const fields_len = zir.extra[extra_index];
        extra_index += 1;
        break :blk fields_len;
    } else 0;

    const decls_len = if (small.has_decls_len) decls_len: {
        const decls_len = zir.extra[extra_index];
        extra_index += 1;
        break :decls_len decls_len;
    } else 0;

    // Skip over decls.
    var decls_it = zir.declIteratorInner(extra_index, decls_len);
    while (decls_it.next()) |_| {}
    extra_index = decls_it.extra_index;

    const body = zir.extra[extra_index..][0..body_len];
    if (fields_len == 0) {
        assert(body.len == 0);
        return;
    }
    extra_index += body.len;

    var decl_arena = decl.value_arena.?.promote(gpa);
    defer decl.value_arena.?.* = decl_arena.state;
    const decl_arena_allocator = decl_arena.allocator();

    var analysis_arena = std.heap.ArenaAllocator.init(gpa);
    defer analysis_arena.deinit();

    var sema: Sema = .{
        .mod = mod,
        .gpa = gpa,
        .arena = analysis_arena.allocator(),
        .perm_arena = decl_arena_allocator,
        .code = zir,
        .owner_decl = decl,
        .func = null,
        .fn_ret_ty = Type.void,
        .owner_func = null,
    };
    defer sema.deinit();

    var wip_captures = try WipCaptureScope.init(gpa, decl_arena_allocator, decl.src_scope);
    defer wip_captures.deinit();

    var block_scope: Block = .{
        .parent = null,
        .sema = &sema,
        .src_decl = decl,
        .namespace = &struct_obj.namespace,
        .wip_capture_scope = wip_captures.scope,
        .instructions = .{},
        .inlining = null,
        .is_comptime = true,
    };
    defer {
        assert(block_scope.instructions.items.len == 0);
        block_scope.params.deinit(gpa);
    }

    if (body.len != 0) {
        try sema.analyzeBody(&block_scope, body);
    }

    try wip_captures.finalize();

    try struct_obj.fields.ensureTotalCapacity(decl_arena_allocator, fields_len);

    const bits_per_field = 4;
    const fields_per_u32 = 32 / bits_per_field;
    const bit_bags_count = std.math.divCeil(usize, fields_len, fields_per_u32) catch unreachable;
    var bit_bag_index: usize = extra_index;
    extra_index += bit_bags_count;
    var cur_bit_bag: u32 = undefined;
    var field_i: u32 = 0;
    while (field_i < fields_len) : (field_i += 1) {
        if (field_i % fields_per_u32 == 0) {
            cur_bit_bag = zir.extra[bit_bag_index];
            bit_bag_index += 1;
        }
        const has_align = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const has_default = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const is_comptime = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const unused = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;

        _ = unused;

        const field_name_zir = zir.nullTerminatedString(zir.extra[extra_index]);
        extra_index += 1;
        const field_type_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
        extra_index += 1;

        // doc_comment
        extra_index += 1;

        // This string needs to outlive the ZIR code.
        const field_name = try decl_arena_allocator.dupe(u8, field_name_zir);
        const field_ty: Type = if (field_type_ref == .none)
            Type.initTag(.noreturn)
        else
            // TODO: if we need to report an error here, use a source location
            // that points to this type expression rather than the struct.
            // But only resolve the source location if we need to emit a compile error.
            try sema.resolveType(&block_scope, src, field_type_ref);

        // TODO emit compile errors for invalid field types
        // such as arrays and pointers inside packed structs.

        if (field_ty.tag() == .generic_poison) {
            return error.GenericPoison;
        }

        const gop = struct_obj.fields.getOrPutAssumeCapacity(field_name);
        assert(!gop.found_existing);
        gop.value_ptr.* = .{
            .ty = try field_ty.copy(decl_arena_allocator),
            .abi_align = 0,
            .default_val = Value.initTag(.unreachable_value),
            .is_comptime = is_comptime,
            .offset = undefined,
        };

        if (has_align) {
            const align_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
            extra_index += 1;
            // TODO: if we need to report an error here, use a source location
            // that points to this alignment expression rather than the struct.
            // But only resolve the source location if we need to emit a compile error.
            gop.value_ptr.abi_align = try sema.resolveAlign(&block_scope, src, align_ref);
        }
        if (has_default) {
            const default_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
            extra_index += 1;
            const default_inst = sema.resolveInst(default_ref);
            // TODO: if we need to report an error here, use a source location
            // that points to this default value expression rather than the struct.
            // But only resolve the source location if we need to emit a compile error.
            const default_val = (try sema.resolveMaybeUndefVal(&block_scope, src, default_inst)) orelse
                return sema.failWithNeededComptime(&block_scope, src);
            gop.value_ptr.default_val = try default_val.copy(decl_arena_allocator);
        }
    }
}

fn semaUnionFields(mod: *Module, union_obj: *Module.Union) CompileError!void {
    const tracy = trace(@src());
    defer tracy.end();

    const gpa = mod.gpa;
    const decl = union_obj.owner_decl;
    const zir = union_obj.namespace.file_scope.zir;
    const extended = zir.instructions.items(.data)[union_obj.zir_index].extended;
    assert(extended.opcode == .union_decl);
    const small = @bitCast(Zir.Inst.UnionDecl.Small, extended.small);
    var extra_index: usize = extended.operand;

    const src: LazySrcLoc = .{ .node_offset = union_obj.node_offset };
    extra_index += @boolToInt(small.has_src_node);

    const tag_type_ref: Zir.Inst.Ref = if (small.has_tag_type) blk: {
        const ty_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
        extra_index += 1;
        break :blk ty_ref;
    } else .none;

    const body_len = if (small.has_body_len) blk: {
        const body_len = zir.extra[extra_index];
        extra_index += 1;
        break :blk body_len;
    } else 0;

    const fields_len = if (small.has_fields_len) blk: {
        const fields_len = zir.extra[extra_index];
        extra_index += 1;
        break :blk fields_len;
    } else 0;

    const decls_len = if (small.has_decls_len) decls_len: {
        const decls_len = zir.extra[extra_index];
        extra_index += 1;
        break :decls_len decls_len;
    } else 0;

    // Skip over decls.
    var decls_it = zir.declIteratorInner(extra_index, decls_len);
    while (decls_it.next()) |_| {}
    extra_index = decls_it.extra_index;

    const body = zir.extra[extra_index..][0..body_len];
    if (fields_len == 0) {
        assert(body.len == 0);
        return;
    }
    extra_index += body.len;

    var decl_arena = union_obj.owner_decl.value_arena.?.promote(gpa);
    defer union_obj.owner_decl.value_arena.?.* = decl_arena.state;
    const decl_arena_allocator = decl_arena.allocator();

    var analysis_arena = std.heap.ArenaAllocator.init(gpa);
    defer analysis_arena.deinit();

    var sema: Sema = .{
        .mod = mod,
        .gpa = gpa,
        .arena = analysis_arena.allocator(),
        .perm_arena = decl_arena_allocator,
        .code = zir,
        .owner_decl = decl,
        .func = null,
        .fn_ret_ty = Type.void,
        .owner_func = null,
    };
    defer sema.deinit();

    var wip_captures = try WipCaptureScope.init(gpa, decl_arena_allocator, decl.src_scope);
    defer wip_captures.deinit();

    var block_scope: Block = .{
        .parent = null,
        .sema = &sema,
        .src_decl = decl,
        .namespace = &union_obj.namespace,
        .wip_capture_scope = wip_captures.scope,
        .instructions = .{},
        .inlining = null,
        .is_comptime = true,
    };
    defer {
        assert(block_scope.instructions.items.len == 0);
        block_scope.params.deinit(gpa);
    }

    if (body.len != 0) {
        try sema.analyzeBody(&block_scope, body);
    }

    try wip_captures.finalize();

    try union_obj.fields.ensureTotalCapacity(decl_arena_allocator, fields_len);

    var int_tag_ty: Type = undefined;
    var enum_field_names: ?*Module.EnumNumbered.NameMap = null;
    var enum_value_map: ?*Module.EnumNumbered.ValueMap = null;
    if (tag_type_ref != .none) {
        const provided_ty = try sema.resolveType(&block_scope, src, tag_type_ref);
        if (small.auto_enum_tag) {
            // The provided type is an integer type and we must construct the enum tag type here.
            int_tag_ty = provided_ty;
            union_obj.tag_ty = try sema.generateUnionTagTypeNumbered(&block_scope, fields_len, provided_ty);
            const enum_obj = union_obj.tag_ty.castTag(.enum_numbered).?.data;
            enum_field_names = &enum_obj.fields;
            enum_value_map = &enum_obj.values;
        } else {
            // The provided type is the enum tag type.
            union_obj.tag_ty = try provided_ty.copy(decl_arena_allocator);
        }
    } else {
        // If auto_enum_tag is false, this is an untagged union. However, for semantic analysis
        // purposes, we still auto-generate an enum tag type the same way. That the union is
        // untagged is represented by the Type tag (union vs union_tagged).
        union_obj.tag_ty = try sema.generateUnionTagTypeSimple(&block_scope, fields_len);
        enum_field_names = &union_obj.tag_ty.castTag(.enum_simple).?.data.fields;
    }

    const target = sema.mod.getTarget();

    const bits_per_field = 4;
    const fields_per_u32 = 32 / bits_per_field;
    const bit_bags_count = std.math.divCeil(usize, fields_len, fields_per_u32) catch unreachable;
    var bit_bag_index: usize = extra_index;
    extra_index += bit_bags_count;
    var cur_bit_bag: u32 = undefined;
    var field_i: u32 = 0;
    var last_tag_val: ?Value = null;
    while (field_i < fields_len) : (field_i += 1) {
        if (field_i % fields_per_u32 == 0) {
            cur_bit_bag = zir.extra[bit_bag_index];
            bit_bag_index += 1;
        }
        const has_type = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const has_align = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const has_tag = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        const unused = @truncate(u1, cur_bit_bag) != 0;
        cur_bit_bag >>= 1;
        _ = unused;

        const field_name_zir = zir.nullTerminatedString(zir.extra[extra_index]);
        extra_index += 1;

        // doc_comment
        extra_index += 1;

        const field_type_ref: Zir.Inst.Ref = if (has_type) blk: {
            const field_type_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
            extra_index += 1;
            break :blk field_type_ref;
        } else .none;

        const align_ref: Zir.Inst.Ref = if (has_align) blk: {
            const align_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
            extra_index += 1;
            break :blk align_ref;
        } else .none;

        const tag_ref: Zir.Inst.Ref = if (has_tag) blk: {
            const tag_ref = @intToEnum(Zir.Inst.Ref, zir.extra[extra_index]);
            extra_index += 1;
            break :blk sema.resolveInst(tag_ref);
        } else .none;

        if (enum_value_map) |map| {
            if (tag_ref != .none) {
                const tag_src = src; // TODO better source location
                const coerced = try sema.coerce(&block_scope, int_tag_ty, tag_ref, tag_src);
                const val = try sema.resolveConstValue(&block_scope, tag_src, coerced);
                last_tag_val = val;

                // This puts the memory into the union arena, not the enum arena, but
                // it is OK since they share the same lifetime.
                const copied_val = try val.copy(decl_arena_allocator);
                map.putAssumeCapacityContext(copied_val, {}, .{
                    .ty = int_tag_ty,
                    .target = target,
                });
            } else {
                const val = if (last_tag_val) |val|
                    try val.intAdd(Value.one, int_tag_ty, sema.arena, target)
                else
                    Value.zero;
                last_tag_val = val;

                const copied_val = try val.copy(decl_arena_allocator);
                map.putAssumeCapacityContext(copied_val, {}, .{
                    .ty = int_tag_ty,
                    .target = target,
                });
            }
        }

        // This string needs to outlive the ZIR code.
        const field_name = try decl_arena_allocator.dupe(u8, field_name_zir);
        if (enum_field_names) |set| {
            set.putAssumeCapacity(field_name, {});
        }

        const field_ty: Type = if (!has_type)
            Type.void
        else if (field_type_ref == .none)
            Type.initTag(.noreturn)
        else
            // TODO: if we need to report an error here, use a source location
            // that points to this type expression rather than the union.
            // But only resolve the source location if we need to emit a compile error.
            try sema.resolveType(&block_scope, src, field_type_ref);

        if (field_ty.tag() == .generic_poison) {
            return error.GenericPoison;
        }

        const gop = union_obj.fields.getOrPutAssumeCapacity(field_name);
        assert(!gop.found_existing);
        gop.value_ptr.* = .{
            .ty = try field_ty.copy(decl_arena_allocator),
            .abi_align = 0,
        };

        if (align_ref != .none) {
            // TODO: if we need to report an error here, use a source location
            // that points to this alignment expression rather than the struct.
            // But only resolve the source location if we need to emit a compile error.
            gop.value_ptr.abi_align = try sema.resolveAlign(&block_scope, src, align_ref);
        } else {
            gop.value_ptr.abi_align = 0;
        }
    }
}

fn generateUnionTagTypeNumbered(
    sema: *Sema,
    block: *Block,
    fields_len: u32,
    int_ty: Type,
) !Type {
    const mod = sema.mod;

    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const enum_obj = try new_decl_arena_allocator.create(Module.EnumNumbered);
    const enum_ty_payload = try new_decl_arena_allocator.create(Type.Payload.EnumNumbered);
    enum_ty_payload.* = .{
        .base = .{ .tag = .enum_numbered },
        .data = enum_obj,
    };
    const enum_ty = Type.initPayload(&enum_ty_payload.base);
    const enum_val = try Value.Tag.ty.create(new_decl_arena_allocator, enum_ty);
    // TODO better type name
    const new_decl = try mod.createAnonymousDecl(block, .{
        .ty = Type.type,
        .val = enum_val,
    });
    new_decl.owns_tv = true;
    errdefer mod.abortAnonDecl(new_decl);

    enum_obj.* = .{
        .owner_decl = new_decl,
        .tag_ty = int_ty,
        .fields = .{},
        .values = .{},
        .node_offset = 0,
    };
    // Here we pre-allocate the maps using the decl arena.
    try enum_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);
    try enum_obj.values.ensureTotalCapacityContext(new_decl_arena_allocator, fields_len, .{
        .ty = int_ty,
        .target = sema.mod.getTarget(),
    });
    try new_decl.finalizeNewArena(&new_decl_arena);
    return enum_ty;
}

fn generateUnionTagTypeSimple(sema: *Sema, block: *Block, fields_len: usize) !Type {
    const mod = sema.mod;

    var new_decl_arena = std.heap.ArenaAllocator.init(sema.gpa);
    errdefer new_decl_arena.deinit();
    const new_decl_arena_allocator = new_decl_arena.allocator();

    const enum_obj = try new_decl_arena_allocator.create(Module.EnumSimple);
    const enum_ty_payload = try new_decl_arena_allocator.create(Type.Payload.EnumSimple);
    enum_ty_payload.* = .{
        .base = .{ .tag = .enum_simple },
        .data = enum_obj,
    };
    const enum_ty = Type.initPayload(&enum_ty_payload.base);
    const enum_val = try Value.Tag.ty.create(new_decl_arena_allocator, enum_ty);
    // TODO better type name
    const new_decl = try mod.createAnonymousDecl(block, .{
        .ty = Type.type,
        .val = enum_val,
    });
    new_decl.owns_tv = true;
    errdefer mod.abortAnonDecl(new_decl);

    enum_obj.* = .{
        .owner_decl = new_decl,
        .fields = .{},
        .node_offset = 0,
    };
    // Here we pre-allocate the maps using the decl arena.
    try enum_obj.fields.ensureTotalCapacity(new_decl_arena_allocator, fields_len);
    try new_decl.finalizeNewArena(&new_decl_arena);
    return enum_ty;
}

fn getBuiltin(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    name: []const u8,
) CompileError!Air.Inst.Ref {
    const mod = sema.mod;
    const std_pkg = mod.main_pkg.table.get("std").?;
    const std_file = (mod.importPkg(std_pkg) catch unreachable).file;
    const opt_builtin_inst = try sema.namespaceLookupRef(
        block,
        src,
        std_file.root_decl.?.src_namespace,
        "builtin",
    );
    const builtin_inst = try sema.analyzeLoad(block, src, opt_builtin_inst.?, src);
    const builtin_ty = try sema.analyzeAsType(block, src, builtin_inst);
    const opt_ty_decl = try sema.namespaceLookup(
        block,
        src,
        builtin_ty.getNamespace().?,
        name,
    );
    return sema.analyzeDeclVal(block, src, opt_ty_decl.?);
}

fn getBuiltinType(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    name: []const u8,
) CompileError!Type {
    const ty_inst = try sema.getBuiltin(block, src, name);
    const result_ty = try sema.analyzeAsType(block, src, ty_inst);
    try sema.queueFullTypeResolution(result_ty);
    return result_ty;
}

/// There is another implementation of this in `Type.onePossibleValue`. This one
/// in `Sema` is for calling during semantic analysis, and performs field resolution
/// to get the answer. The one in `Type` is for calling during codegen and asserts
/// that the types are already resolved.
/// TODO assert the return value matches `ty.onePossibleValue`
pub fn typeHasOnePossibleValue(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    ty: Type,
) CompileError!?Value {
    switch (ty.tag()) {
        .f16,
        .f32,
        .f64,
        .f80,
        .f128,
        .c_longdouble,
        .comptime_int,
        .comptime_float,
        .u1,
        .u8,
        .i8,
        .u16,
        .i16,
        .u32,
        .i32,
        .u64,
        .i64,
        .u128,
        .i128,
        .usize,
        .isize,
        .c_short,
        .c_ushort,
        .c_int,
        .c_uint,
        .c_long,
        .c_ulong,
        .c_longlong,
        .c_ulonglong,
        .bool,
        .type,
        .anyerror,
        .fn_noreturn_no_args,
        .fn_void_no_args,
        .fn_naked_noreturn_no_args,
        .fn_ccc_void_no_args,
        .function,
        .single_const_pointer_to_comptime_int,
        .array_sentinel,
        .array_u8_sentinel_0,
        .const_slice_u8,
        .const_slice_u8_sentinel_0,
        .const_slice,
        .mut_slice,
        .anyopaque,
        .optional,
        .optional_single_mut_pointer,
        .optional_single_const_pointer,
        .enum_literal,
        .anyerror_void_error_union,
        .error_union,
        .error_set,
        .error_set_single,
        .error_set_inferred,
        .error_set_merged,
        .@"opaque",
        .var_args_param,
        .manyptr_u8,
        .manyptr_const_u8,
        .manyptr_const_u8_sentinel_0,
        .atomic_order,
        .atomic_rmw_op,
        .calling_convention,
        .address_space,
        .float_mode,
        .reduce_op,
        .call_options,
        .prefetch_options,
        .export_options,
        .extern_options,
        .type_info,
        .@"anyframe",
        .anyframe_T,
        .many_const_pointer,
        .many_mut_pointer,
        .c_const_pointer,
        .c_mut_pointer,
        .single_const_pointer,
        .single_mut_pointer,
        .pointer,
        .bound_fn,
        => return null,

        .@"struct" => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const s = resolved_ty.castTag(.@"struct").?.data;
            for (s.fields.values()) |value| {
                if (value.is_comptime) continue;
                if ((try sema.typeHasOnePossibleValue(block, src, value.ty)) == null) {
                    return null;
                }
            }
            return Value.initTag(.empty_struct_value);
        },

        .tuple, .anon_struct => {
            const tuple = ty.tupleFields();
            for (tuple.values) |val| {
                if (val.tag() == .unreachable_value) {
                    return null; // non-comptime field
                }
            }
            return Value.initTag(.empty_struct_value);
        },

        .enum_numbered => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const enum_obj = resolved_ty.castTag(.enum_numbered).?.data;
            if (enum_obj.fields.count() == 1) {
                if (enum_obj.values.count() == 0) {
                    return Value.zero; // auto-numbered
                } else {
                    return enum_obj.values.keys()[0];
                }
            } else {
                return null;
            }
        },
        .enum_full => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const enum_obj = resolved_ty.castTag(.enum_full).?.data;
            if (enum_obj.fields.count() == 1) {
                if (enum_obj.values.count() == 0) {
                    return Value.zero; // auto-numbered
                } else {
                    return enum_obj.values.keys()[0];
                }
            } else {
                return null;
            }
        },
        .enum_simple => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const enum_simple = resolved_ty.castTag(.enum_simple).?.data;
            if (enum_simple.fields.count() == 1) {
                return Value.zero;
            } else {
                return null;
            }
        },
        .enum_nonexhaustive => {
            const tag_ty = ty.castTag(.enum_nonexhaustive).?.data.tag_ty;
            if (!(try sema.typeHasRuntimeBits(block, src, tag_ty))) {
                return Value.zero;
            } else {
                return null;
            }
        },
        .@"union", .union_tagged => {
            const resolved_ty = try sema.resolveTypeFields(block, src, ty);
            const union_obj = resolved_ty.cast(Type.Payload.Union).?.data;
            const tag_val = (try sema.typeHasOnePossibleValue(block, src, union_obj.tag_ty)) orelse
                return null;
            const only_field = union_obj.fields.values()[0];
            const val_val = (try sema.typeHasOnePossibleValue(block, src, only_field.ty)) orelse
                return null;
            // TODO make this not allocate. The function in `Type.onePossibleValue`
            // currently returns `empty_struct_value` and we should do that here too.
            return try Value.Tag.@"union".create(sema.arena, .{
                .tag = tag_val,
                .val = val_val,
            });
        },

        .empty_struct, .empty_struct_literal => return Value.initTag(.empty_struct_value),
        .void => return Value.void,
        .noreturn => return Value.initTag(.unreachable_value),
        .@"null" => return Value.@"null",
        .@"undefined" => return Value.initTag(.undef),

        .int_unsigned, .int_signed => {
            if (ty.cast(Type.Payload.Bits).?.data == 0) {
                return Value.zero;
            } else {
                return null;
            }
        },
        .vector, .array, .array_u8 => {
            if (ty.arrayLen() == 0)
                return Value.initTag(.empty_array);
            if ((try sema.typeHasOnePossibleValue(block, src, ty.elemType())) != null) {
                return Value.initTag(.the_only_possible_value);
            }
            return null;
        },

        .inferred_alloc_const => unreachable,
        .inferred_alloc_mut => unreachable,
        .generic_poison => return error.GenericPoison,
    }
}

fn getAstTree(sema: *Sema, block: *Block) CompileError!*const std.zig.Ast {
    return block.namespace.file_scope.getTree(sema.gpa) catch |err| {
        log.err("unable to load AST to report compile error: {s}", .{@errorName(err)});
        return error.AnalysisFail;
    };
}

fn enumFieldSrcLoc(
    decl: *Decl,
    tree: std.zig.Ast,
    node_offset: i32,
    field_index: usize,
) LazySrcLoc {
    @setCold(true);
    const enum_node = decl.relativeToNodeIndex(node_offset);
    const node_tags = tree.nodes.items(.tag);
    var buffer: [2]std.zig.Ast.Node.Index = undefined;
    const container_decl = switch (node_tags[enum_node]) {
        .container_decl,
        .container_decl_trailing,
        => tree.containerDecl(enum_node),

        .container_decl_two,
        .container_decl_two_trailing,
        => tree.containerDeclTwo(&buffer, enum_node),

        .container_decl_arg,
        .container_decl_arg_trailing,
        => tree.containerDeclArg(enum_node),

        else => unreachable,
    };
    var it_index: usize = 0;
    for (container_decl.ast.members) |member_node| {
        switch (node_tags[member_node]) {
            .container_field_init,
            .container_field_align,
            .container_field,
            => {
                if (it_index == field_index) {
                    return .{ .node_offset = decl.nodeIndexToRelative(member_node) };
                }
                it_index += 1;
            },

            else => continue,
        }
    } else unreachable;
}

/// Returns the type of the AIR instruction.
fn typeOf(sema: *Sema, inst: Air.Inst.Ref) Type {
    return sema.getTmpAir().typeOf(inst);
}

pub fn getTmpAir(sema: Sema) Air {
    return .{
        .instructions = sema.air_instructions.slice(),
        .extra = sema.air_extra.items,
        .values = sema.air_values.items,
    };
}

pub fn addType(sema: *Sema, ty: Type) !Air.Inst.Ref {
    switch (ty.tag()) {
        .u1 => return .u1_type,
        .u8 => return .u8_type,
        .i8 => return .i8_type,
        .u16 => return .u16_type,
        .i16 => return .i16_type,
        .u32 => return .u32_type,
        .i32 => return .i32_type,
        .u64 => return .u64_type,
        .i64 => return .i64_type,
        .u128 => return .u128_type,
        .i128 => return .i128_type,
        .usize => return .usize_type,
        .isize => return .isize_type,
        .c_short => return .c_short_type,
        .c_ushort => return .c_ushort_type,
        .c_int => return .c_int_type,
        .c_uint => return .c_uint_type,
        .c_long => return .c_long_type,
        .c_ulong => return .c_ulong_type,
        .c_longlong => return .c_longlong_type,
        .c_ulonglong => return .c_ulonglong_type,
        .c_longdouble => return .c_longdouble_type,
        .f16 => return .f16_type,
        .f32 => return .f32_type,
        .f64 => return .f64_type,
        .f80 => return .f80_type,
        .f128 => return .f128_type,
        .anyopaque => return .anyopaque_type,
        .bool => return .bool_type,
        .void => return .void_type,
        .type => return .type_type,
        .anyerror => return .anyerror_type,
        .comptime_int => return .comptime_int_type,
        .comptime_float => return .comptime_float_type,
        .noreturn => return .noreturn_type,
        .@"anyframe" => return .anyframe_type,
        .@"null" => return .null_type,
        .@"undefined" => return .undefined_type,
        .enum_literal => return .enum_literal_type,
        .atomic_order => return .atomic_order_type,
        .atomic_rmw_op => return .atomic_rmw_op_type,
        .calling_convention => return .calling_convention_type,
        .address_space => return .address_space_type,
        .float_mode => return .float_mode_type,
        .reduce_op => return .reduce_op_type,
        .call_options => return .call_options_type,
        .prefetch_options => return .prefetch_options_type,
        .export_options => return .export_options_type,
        .extern_options => return .extern_options_type,
        .type_info => return .type_info_type,
        .manyptr_u8 => return .manyptr_u8_type,
        .manyptr_const_u8 => return .manyptr_const_u8_type,
        .fn_noreturn_no_args => return .fn_noreturn_no_args_type,
        .fn_void_no_args => return .fn_void_no_args_type,
        .fn_naked_noreturn_no_args => return .fn_naked_noreturn_no_args_type,
        .fn_ccc_void_no_args => return .fn_ccc_void_no_args_type,
        .single_const_pointer_to_comptime_int => return .single_const_pointer_to_comptime_int_type,
        .const_slice_u8 => return .const_slice_u8_type,
        .anyerror_void_error_union => return .anyerror_void_error_union_type,
        .generic_poison => return .generic_poison_type,
        else => {},
    }
    try sema.air_instructions.append(sema.gpa, .{
        .tag = .const_ty,
        .data = .{ .ty = ty },
    });
    return Air.indexToRef(@intCast(u32, sema.air_instructions.len - 1));
}

fn addIntUnsigned(sema: *Sema, ty: Type, int: u64) CompileError!Air.Inst.Ref {
    return sema.addConstant(ty, try Value.Tag.int_u64.create(sema.arena, int));
}

fn addConstUndef(sema: *Sema, ty: Type) CompileError!Air.Inst.Ref {
    return sema.addConstant(ty, Value.undef);
}

pub fn addConstant(sema: *Sema, ty: Type, val: Value) SemaError!Air.Inst.Ref {
    const gpa = sema.gpa;
    const ty_inst = try sema.addType(ty);
    try sema.air_values.append(gpa, val);
    try sema.air_instructions.append(gpa, .{
        .tag = .constant,
        .data = .{ .ty_pl = .{
            .ty = ty_inst,
            .payload = @intCast(u32, sema.air_values.items.len - 1),
        } },
    });
    return Air.indexToRef(@intCast(u32, sema.air_instructions.len - 1));
}

pub fn addExtra(sema: *Sema, extra: anytype) Allocator.Error!u32 {
    const fields = std.meta.fields(@TypeOf(extra));
    try sema.air_extra.ensureUnusedCapacity(sema.gpa, fields.len);
    return addExtraAssumeCapacity(sema, extra);
}

pub fn addExtraAssumeCapacity(sema: *Sema, extra: anytype) u32 {
    const fields = std.meta.fields(@TypeOf(extra));
    const result = @intCast(u32, sema.air_extra.items.len);
    inline for (fields) |field| {
        sema.air_extra.appendAssumeCapacity(switch (field.field_type) {
            u32 => @field(extra, field.name),
            Air.Inst.Ref => @enumToInt(@field(extra, field.name)),
            i32 => @bitCast(u32, @field(extra, field.name)),
            else => @compileError("bad field type"),
        });
    }
    return result;
}

fn appendRefsAssumeCapacity(sema: *Sema, refs: []const Air.Inst.Ref) void {
    const coerced = @bitCast([]const u32, refs);
    sema.air_extra.appendSliceAssumeCapacity(coerced);
}

fn getBreakBlock(sema: *Sema, inst_index: Air.Inst.Index) ?Air.Inst.Index {
    const air_datas = sema.air_instructions.items(.data);
    const air_tags = sema.air_instructions.items(.tag);
    switch (air_tags[inst_index]) {
        .br => return air_datas[inst_index].br.block_inst,
        else => return null,
    }
}

fn isComptimeKnown(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    inst: Air.Inst.Ref,
) !bool {
    return (try sema.resolveMaybeUndefVal(block, src, inst)) != null;
}

fn analyzeComptimeAlloc(
    sema: *Sema,
    block: *Block,
    var_type: Type,
    alignment: u32,
    src: LazySrcLoc,
) CompileError!Air.Inst.Ref {
    // Needed to make an anon decl with type `var_type` (the `finish()` call below).
    _ = try sema.typeHasOnePossibleValue(block, src, var_type);

    const target = sema.mod.getTarget();
    const ptr_type = try Type.ptr(sema.arena, target, .{
        .pointee_type = var_type,
        .@"addrspace" = target_util.defaultAddressSpace(sema.mod.getTarget(), .global_constant),
        .@"align" = alignment,
    });

    var anon_decl = try block.startAnonDecl(src);
    defer anon_decl.deinit();

    const decl = try anon_decl.finish(
        try var_type.copy(anon_decl.arena()),
        // There will be stores before the first load, but they may be to sub-elements or
        // sub-fields. So we need to initialize with undef to allow the mechanism to expand
        // into fields/elements and have those overridden with stored values.
        Value.undef,
        alignment,
    );
    decl.@"align" = alignment;

    try sema.mod.declareDeclDependency(sema.owner_decl, decl);
    return sema.addConstant(ptr_type, try Value.Tag.decl_ref_mut.create(sema.arena, .{
        .runtime_index = block.runtime_index,
        .decl = decl,
    }));
}

/// The places where a user can specify an address space attribute
pub const AddressSpaceContext = enum {
    /// A function is specified to be placed in a certain address space.
    function,

    /// A (global) variable is specified to be placed in a certain address space.
    /// In contrast to .constant, these values (and thus the address space they will be
    /// placed in) are required to be mutable.
    variable,

    /// A (global) constant value is specified to be placed in a certain address space.
    /// In contrast to .variable, values placed in this address space are not required to be mutable.
    constant,

    /// A pointer is ascripted to point into a certain address space.
    pointer,
};

pub fn analyzeAddrspace(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    zir_ref: Zir.Inst.Ref,
    ctx: AddressSpaceContext,
) !std.builtin.AddressSpace {
    const addrspace_tv = try sema.resolveInstConst(block, src, zir_ref);
    const address_space = addrspace_tv.val.toEnum(std.builtin.AddressSpace);
    const target = sema.mod.getTarget();
    const arch = target.cpu.arch;
    const is_gpu = arch == .nvptx or arch == .nvptx64;

    const supported = switch (address_space) {
        .generic => true,
        .gs, .fs, .ss => (arch == .i386 or arch == .x86_64) and ctx == .pointer,
        // TODO: check that .shared and .local are left uninitialized
        .global, .param, .shared, .local => is_gpu,
        .constant => is_gpu and (ctx == .constant),
    };

    if (!supported) {
        // TODO error messages could be made more elaborate here
        const entity = switch (ctx) {
            .function => "functions",
            .variable => "mutable values",
            .constant => "constant values",
            .pointer => "pointers",
        };
        return sema.fail(
            block,
            src,
            "{s} with address space '{s}' are not supported on {s}",
            .{ entity, @tagName(address_space), arch.genericName() },
        );
    }

    return address_space;
}

/// Asserts the value is a pointer and dereferences it.
/// Returns `null` if the pointer contents cannot be loaded at comptime.
fn pointerDeref(sema: *Sema, block: *Block, src: LazySrcLoc, ptr_val: Value, ptr_ty: Type) CompileError!?Value {
    const load_ty = ptr_ty.childType();
    const target = sema.mod.getTarget();
    const deref = sema.beginComptimePtrLoad(block, src, ptr_val, load_ty) catch |err| switch (err) {
        error.RuntimeLoad => return null,
        else => |e| return e,
    };

    if (deref.pointee) |tv| {
        const coerce_in_mem_ok =
            (try sema.coerceInMemoryAllowed(block, load_ty, tv.ty, false, target, src, src)) == .ok or
            (try sema.coerceInMemoryAllowed(block, tv.ty, load_ty, false, target, src, src)) == .ok;
        if (coerce_in_mem_ok) {
            // We have a Value that lines up in virtual memory exactly with what we want to load,
            // and it is in-memory coercible to load_ty. It may be returned without modifications.
            if (deref.is_mutable) {
                // The decl whose value we are obtaining here may be overwritten with
                // a different value upon further semantic analysis, which would
                // invalidate this memory. So we must copy here.
                return try tv.val.copy(sema.arena);
            }
            return tv.val;
        }
    }

    // The type is not in-memory coercible or the direct dereference failed, so it must
    // be bitcast according to the pointer type we are performing the load through.
    if (!load_ty.hasWellDefinedLayout())
        return sema.fail(block, src, "comptime dereference requires {} to have a well-defined layout, but it does not.", .{load_ty.fmt(target)});

    const load_sz = try sema.typeAbiSize(block, src, load_ty);

    // Try the smaller bit-cast first, since that's more efficient than using the larger `parent`
    if (deref.pointee) |tv| if (load_sz <= try sema.typeAbiSize(block, src, tv.ty))
        return try sema.bitCastVal(block, src, tv.val, tv.ty, load_ty, 0);

    // If that fails, try to bit-cast from the largest parent value with a well-defined layout
    if (deref.parent) |parent| if (load_sz + parent.byte_offset <= try sema.typeAbiSize(block, src, parent.tv.ty))
        return try sema.bitCastVal(block, src, parent.tv.val, parent.tv.ty, load_ty, parent.byte_offset);

    if (deref.ty_without_well_defined_layout) |bad_ty| {
        // We got no parent for bit-casting, or the parent we got was too small. Either way, the problem
        // is that some type we encountered when de-referencing does not have a well-defined layout.
        return sema.fail(block, src, "comptime dereference requires {} to have a well-defined layout, but it does not.", .{bad_ty.fmt(target)});
    } else {
        // If all encountered types had well-defined layouts, the parent is the root decl and it just
        // wasn't big enough for the load.
        return sema.fail(block, src, "dereference of {} exceeds bounds of containing decl of type {}", .{ ptr_ty.fmt(target), deref.parent.?.tv.ty.fmt(target) });
    }
}

/// Used to convert a u64 value to a usize value, emitting a compile error if the number
/// is too big to fit.
fn usizeCast(sema: *Sema, block: *Block, src: LazySrcLoc, int: u64) CompileError!usize {
    if (@bitSizeOf(u64) <= @bitSizeOf(usize)) return int;
    return std.math.cast(usize, int) catch |err| switch (err) {
        error.Overflow => return sema.fail(block, src, "expression produces integer value {d} which is too big for this compiler implementation to handle", .{int}),
    };
}

/// For pointer-like optionals, it returns the pointer type. For pointers,
/// the type is returned unmodified.
/// This can return `error.AnalysisFail` because it sometimes requires resolving whether
/// a type has zero bits, which can cause a "foo depends on itself" compile error.
/// This logic must be kept in sync with `Type.isPtrLikeOptional`.
fn typePtrOrOptionalPtrTy(
    sema: *Sema,
    block: *Block,
    ty: Type,
    buf: *Type.Payload.ElemType,
    src: LazySrcLoc,
) !?Type {
    switch (ty.tag()) {
        .optional_single_const_pointer,
        .optional_single_mut_pointer,
        .c_const_pointer,
        .c_mut_pointer,
        => return ty.optionalChild(buf),

        .single_const_pointer_to_comptime_int,
        .single_const_pointer,
        .single_mut_pointer,
        .many_const_pointer,
        .many_mut_pointer,
        .manyptr_u8,
        .manyptr_const_u8,
        .manyptr_const_u8_sentinel_0,
        => return ty,

        .pointer => switch (ty.ptrSize()) {
            .Slice => return null,
            .C => return ty.optionalChild(buf),
            else => return ty,
        },

        .inferred_alloc_const => unreachable,
        .inferred_alloc_mut => unreachable,

        .optional => {
            const child_type = ty.optionalChild(buf);
            if (child_type.zigTypeTag() != .Pointer) return null;

            const info = child_type.ptrInfo().data;
            switch (info.size) {
                .Slice, .C => return null,
                .Many, .One => {
                    if (info.@"allowzero") return null;

                    // optionals of zero sized types behave like bools, not pointers
                    if ((try sema.typeHasOnePossibleValue(block, src, child_type)) != null) {
                        return null;
                    }

                    return child_type;
                },
            }
        },

        else => return null,
    }
}

/// `generic_poison` will return false.
/// This function returns false negatives when structs and unions are having their
/// field types resolved.
/// TODO assert the return value matches `ty.comptimeOnly`
/// TODO merge these implementations together with the "advanced"/sema_kit pattern seen
/// elsewhere in value.zig
pub fn typeRequiresComptime(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!bool {
    if (build_options.omit_stage2)
        @panic("sadly stage2 is omitted from this build to save memory on the CI server");
    return switch (ty.tag()) {
        .u1,
        .u8,
        .i8,
        .u16,
        .i16,
        .u32,
        .i32,
        .u64,
        .i64,
        .u128,
        .i128,
        .usize,
        .isize,
        .c_short,
        .c_ushort,
        .c_int,
        .c_uint,
        .c_long,
        .c_ulong,
        .c_longlong,
        .c_ulonglong,
        .c_longdouble,
        .f16,
        .f32,
        .f64,
        .f80,
        .f128,
        .anyopaque,
        .bool,
        .void,
        .anyerror,
        .noreturn,
        .@"anyframe",
        .@"null",
        .@"undefined",
        .atomic_order,
        .atomic_rmw_op,
        .calling_convention,
        .address_space,
        .float_mode,
        .reduce_op,
        .call_options,
        .prefetch_options,
        .export_options,
        .extern_options,
        .manyptr_u8,
        .manyptr_const_u8,
        .manyptr_const_u8_sentinel_0,
        .const_slice_u8,
        .const_slice_u8_sentinel_0,
        .anyerror_void_error_union,
        .empty_struct_literal,
        .empty_struct,
        .error_set,
        .error_set_single,
        .error_set_inferred,
        .error_set_merged,
        .@"opaque",
        .generic_poison,
        .array_u8,
        .array_u8_sentinel_0,
        .int_signed,
        .int_unsigned,
        .enum_simple,
        => false,

        .single_const_pointer_to_comptime_int,
        .type,
        .comptime_int,
        .comptime_float,
        .enum_literal,
        .type_info,
        // These are function bodies, not function pointers.
        .fn_noreturn_no_args,
        .fn_void_no_args,
        .fn_naked_noreturn_no_args,
        .fn_ccc_void_no_args,
        .function,
        => true,

        .var_args_param => unreachable,
        .inferred_alloc_mut => unreachable,
        .inferred_alloc_const => unreachable,
        .bound_fn => unreachable,

        .array,
        .array_sentinel,
        .vector,
        => return sema.typeRequiresComptime(block, src, ty.childType()),

        .pointer,
        .single_const_pointer,
        .single_mut_pointer,
        .many_const_pointer,
        .many_mut_pointer,
        .c_const_pointer,
        .c_mut_pointer,
        .const_slice,
        .mut_slice,
        => {
            const child_ty = ty.childType();
            if (child_ty.zigTypeTag() == .Fn) {
                return false;
            } else {
                return sema.typeRequiresComptime(block, src, child_ty);
            }
        },

        .optional,
        .optional_single_mut_pointer,
        .optional_single_const_pointer,
        => {
            var buf: Type.Payload.ElemType = undefined;
            return sema.typeRequiresComptime(block, src, ty.optionalChild(&buf));
        },

        .tuple, .anon_struct => {
            const tuple = ty.tupleFields();
            for (tuple.types) |field_ty, i| {
                const have_comptime_val = tuple.values[i].tag() != .unreachable_value;
                if (!have_comptime_val and try sema.typeRequiresComptime(block, src, field_ty)) {
                    return true;
                }
            }
            return false;
        },

        .@"struct" => {
            const struct_obj = ty.castTag(.@"struct").?.data;
            switch (struct_obj.requires_comptime) {
                .no, .wip => return false,
                .yes => return true,
                .unknown => {
                    if (struct_obj.status == .field_types_wip)
                        return false;

                    try sema.resolveTypeFieldsStruct(block, src, ty, struct_obj);

                    struct_obj.requires_comptime = .wip;
                    for (struct_obj.fields.values()) |field| {
                        if (field.is_comptime) continue;
                        if (try sema.typeRequiresComptime(block, src, field.ty)) {
                            struct_obj.requires_comptime = .yes;
                            return true;
                        }
                    }
                    struct_obj.requires_comptime = .no;
                    return false;
                },
            }
        },

        .@"union", .union_tagged => {
            const union_obj = ty.cast(Type.Payload.Union).?.data;
            switch (union_obj.requires_comptime) {
                .no, .wip => return false,
                .yes => return true,
                .unknown => {
                    if (union_obj.status == .field_types_wip)
                        return false;

                    try sema.resolveTypeFieldsUnion(block, src, ty, union_obj);

                    union_obj.requires_comptime = .wip;
                    for (union_obj.fields.values()) |field| {
                        if (try sema.typeRequiresComptime(block, src, field.ty)) {
                            union_obj.requires_comptime = .yes;
                            return true;
                        }
                    }
                    union_obj.requires_comptime = .no;
                    return false;
                },
            }
        },

        .error_union => return sema.typeRequiresComptime(block, src, ty.errorUnionPayload()),
        .anyframe_T => {
            const child_ty = ty.castTag(.anyframe_T).?.data;
            return sema.typeRequiresComptime(block, src, child_ty);
        },
        .enum_numbered => {
            const tag_ty = ty.castTag(.enum_numbered).?.data.tag_ty;
            return sema.typeRequiresComptime(block, src, tag_ty);
        },
        .enum_full, .enum_nonexhaustive => {
            const tag_ty = ty.cast(Type.Payload.EnumFull).?.data.tag_ty;
            return sema.typeRequiresComptime(block, src, tag_ty);
        },
    };
}

pub fn typeHasRuntimeBits(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!bool {
    if ((try sema.typeHasOnePossibleValue(block, src, ty)) != null) return false;
    if (try sema.typeRequiresComptime(block, src, ty)) return false;
    return true;
}

fn typeAbiSize(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) !u64 {
    try sema.resolveTypeLayout(block, src, ty);
    const target = sema.mod.getTarget();
    return ty.abiSize(target);
}

fn typeAbiAlignment(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!u32 {
    const target = sema.mod.getTarget();
    return (try ty.abiAlignmentAdvanced(target, .{ .sema_kit = sema.kit(block, src) })).scalar;
}

/// Not valid to call for packed unions.
/// Keep implementation in sync with `Module.Union.Field.normalAlignment`.
fn unionFieldAlignment(
    sema: *Sema,
    block: *Block,
    src: LazySrcLoc,
    field: Module.Union.Field,
) !u32 {
    if (field.abi_align == 0) {
        return sema.typeAbiAlignment(block, src, field.ty);
    } else {
        return field.abi_align;
    }
}

/// Synchronize logic with `Type.isFnOrHasRuntimeBits`.
pub fn fnHasRuntimeBits(sema: *Sema, block: *Block, src: LazySrcLoc, ty: Type) CompileError!bool {
    const fn_info = ty.fnInfo();
    if (fn_info.is_generic) return false;
    if (fn_info.is_var_args) return true;
    switch (fn_info.cc) {
        // If there was a comptime calling convention, it should also return false here.
        .Inline => return false,
        else => {},
    }
    if (try sema.typeRequiresComptime(block, src, fn_info.return_type)) {
        return false;
    }
    return true;
}

fn unionFieldIndex(
    sema: *Sema,
    block: *Block,
    unresolved_union_ty: Type,
    field_name: []const u8,
    field_src: LazySrcLoc,
) !u32 {
    const union_ty = try sema.resolveTypeFields(block, field_src, unresolved_union_ty);
    const union_obj = union_ty.cast(Type.Payload.Union).?.data;
    const field_index_usize = union_obj.fields.getIndex(field_name) orelse
        return sema.failWithBadUnionFieldAccess(block, union_obj, field_src, field_name);
    return @intCast(u32, field_index_usize);
}

fn structFieldIndex(
    sema: *Sema,
    block: *Block,
    unresolved_struct_ty: Type,
    field_name: []const u8,
    field_src: LazySrcLoc,
) !u32 {
    const struct_ty = try sema.resolveTypeFields(block, field_src, unresolved_struct_ty);
    const struct_obj = struct_ty.castTag(.@"struct").?.data;
    const field_index_usize = struct_obj.fields.getIndex(field_name) orelse
        return sema.failWithBadStructFieldAccess(block, struct_obj, field_src, field_name);
    return @intCast(u32, field_index_usize);
}

fn anonStructFieldIndex(
    sema: *Sema,
    block: *Block,
    struct_ty: Type,
    field_name: []const u8,
    field_src: LazySrcLoc,
) !u32 {
    const anon_struct = struct_ty.castTag(.anon_struct).?.data;
    for (anon_struct.names) |name, i| {
        if (mem.eql(u8, name, field_name)) {
            return @intCast(u32, i);
        }
    }
    const target = sema.mod.getTarget();
    return sema.fail(block, field_src, "anonymous struct {} has no such field '{s}'", .{
        struct_ty.fmt(target), field_name,
    });
}

fn kit(sema: *Sema, block: *Block, src: LazySrcLoc) Module.WipAnalysis {
    return .{ .sema = sema, .block = block, .src = src };
}

fn queueFullTypeResolution(sema: *Sema, ty: Type) !void {
    const inst_ref = try sema.addType(ty);
    try sema.types_to_resolve.append(sema.gpa, inst_ref);
}
