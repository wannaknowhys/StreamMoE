# PATCH 拆分踩坑记录（2026-08-28）

拆分 `patches/route-b-inject.patch`（原含 route B 核心 + spec 统计 + GGUF alignment + export 参数杂质）
为 4 个主题 patch（`route-b-inject` / `spec-stats` / `gguf-alignment` / `export-args`）时的**实测结论**。
生成脚本：`temp/split_routeb.js`（输出 `temp/patches_split/`，已复制进 `patches/`）。

## git apply 的 hunk 行为（全部实测确认）

1. **hunk 必须带尾部上下文**：
   - 只有头部上下文 + 纯新增（无尾）→ `apply` **失败**（`patch failed: <file>:<line>`）。
   - 纯新增 + 尾部上下文（无头）→ `apply` **OK**（`@@ -1686,3` 行号 + 尾 3 行定位）。
   - 头 + 尾上下文 → `apply` **OK**。
   拆分同文件 hunk 时，被拆出的"插入段"必须保留尾部上下文作为定位锚点；头部上下文可省。

2. **同 patch 同文件多 hunk：行号用 clean 值**。git apply 会对同一 patch 内同一文件的后续 hunk
   **自动累计**前面 hunk 的净增偏移（原 patch 能 apply 全靠这个）。**不要手动加同 patch 的偏移**
   （会 double，`--verbose` 会看到奇怪 offset）。

3. **跨 patch：行号必须手动重算**。后应用 patch 的 hunk 行号 = 原 clean 行号 + 前面所有 patch 对
   同一文件在"该 hunk 位置之前"（按原 `oldStart` 过滤）的净增。这是拆分算法里唯一需要手动的偏移。

## 拆分脚本 / 工具链坑

4. **classify 返回类型必须一致**：`['route-b', h]` 被 `for...of` 遍历时会把字符串 `'route-b'`
   拆成单个字符（`target='r'`）→ `plan['r']` undefined。统一返回数组的数组 `[['route-b', h]]`。

5. **PowerShell `$P` 与 `$p` 大小写不敏感**：`foreach($p in @(...))` 会覆盖外层 `$P` → 路径拼接错乱
   （`route-b-inject.patch/route-b-inject.patch`）。循环变量避开大写名（用 `$PDIR`）。

6. **PowerShell 里 `node -e "..."` 的 `%d`**：PS 把 `%` 当取余运算符 → 命令直接报错。写临时
   js/patch 文件（用 Write 工具）再执行，避免内联转义地狱。

7. **CRLF 检查**：本次原 patch 是 LF-only（`fs.readFileSync` 后无需 `\r` 处理）。如果源文件是
   CRLF，`split('\n')` 会残留 `\r` 导致上下文不匹配——先用 `node -e` 检测 `\r\n` 再决定。

8. **`git apply --verbose` 的 "Hunk #N succeeded at M (offset -K lines)"**：`K` 是 git 自己的 fuzz
   偏移；如果出现非预期 offset，说明手动行号与 git apply 的自动累计叠加错了。

## 验证方法（拆分后必须做）

9. temp clone 子模块到 clean 基线（`git clone --no-checkout` + `checkout <baseline>`）→ 按顺序
   `git apply --check` 每个 patch → 全部应用 → `git diff <patch-state-commit>` 应为**空**。
   本次：4 拆分 patch + `prefill-export-llama.patch` 顺序应用 == `ffe029953`（route B + prefill
   全量状态），逐字节一致，证明拆分无损。

10. hunk 上下文不匹配时，用 `git show HEAD:<file>` 与生成 patch 的上下文行做**逐字符 JSON 对比**
    （`JSON.stringify` 看前导空格数），别靠肉眼。

## 验证仓库命令速记

```bat
git clone --no-checkout third_party/llama.cpp temp/llama_verify
git -C temp/llama_verify checkout f280b2698
git -C temp/llama_verify apply F:\Dev\StreamMoE\temp\patches_split\route-b-inject.patch
rem ... 按顺序 apply 其余 ...
git -C temp/llama_verify diff --stat ffe029953   rem 应为空
```

## Patch 更新：把"B 应用后的后续修改"并进 B.patch（实测 2026-08-29，好用）

问题：应用 A.patch、B.patch 后又在工作区改了代码，想把后续修改并入 B.patch。

方法（A/B 分开 commit -> 修复 amend 进 B -> 重新导出）：
1. 先确保当前工作区状态已提交（含 A+B+后续修改）；未提交则先 commit 或 stash。
2. 临时分支从干净基线重建 A 和 B：
   ```
   git -C temp/llama_verify checkout f280b2698
   git -C temp/llama_verify checkout -b patch-export
   git -C temp/llama_verify apply A.patch ... && git commit -m "A"     rem commit A
   git -C temp/llama_verify apply B.patch ... && git commit -m "B"     rem commit B
   ```
3. 生成后续修改的增量 diff（相对"B 应用后的状态"）：
   ```
   cmd /c "git -C third_party/llama.cpp diff <A+B状态commit> <修复后commit> -- <改动文件> > temp\fix.diff"
   ```
   在临时分支 apply 该 diff -> `git add -A && git commit --amend`（并入 B commit）。
4. 导出新 B.patch：
   - format-patch（含 email 头）：`git format-patch -1 --stdout > B.patch`
   - 纯 diff（patches/ 风格，推荐）：`git -C temp/llama_verify diff <A commit> <B commit> > B.patch`
5. 验证：temp 从基线依次 apply A + 新 B -> `git diff <目标状态>` 应只差非 patch 文件
   （如 tsc_timer.h 这类独立提交的文件），其余一致。
