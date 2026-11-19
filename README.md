# sd-webui-preset-model-matcher

按 Forge Neo 当前 **UI Preset** 过滤 **Checkpoint** 下拉列表的扩展。切换 Preset 后，下拉框只保留与当前架构兼容的模型，并自动选中该 Preset 上次使用的模型（如果它仍然兼容）。

- 不修改 WebUI 源码，全部通过 `script_callbacks` + 运行时 Hook 实现。
- 只读取 safetensors 头部元数据（`__metadata__`）和 Civitai 侧车 JSON，不加载权重。
- 带磁盘缓存，第二次切换 Preset 基本无感。
- 支持 Forge 自带 Checkpoint 元数据编辑器里的 **Model Arch** 手动覆盖。
- 对文件名无法识别的模型，自动读取 `<模型>.civitai.info` / `.civitai.json` 的 `baseModel`。
- SD2 / SD3 被明确排除：Forge Neo 已移除这两个 Preset。

## 安装

把整个目录放到：

```text
sd-webui-forge-classic/extensions/sd-webui-preset-model-matcher/
```

目录结构：

```text
sd-webui-preset-model-matcher/
├─ README.md
├─ .gitignore
├─ scripts/
│  ├─ preset_model_matcher.py   # 核心：架构识别、缓存、过滤
│  └─ ui_binding.py             # Forge Hook：刷新与 Preset 切换联动
└─ storage/                     # 运行时自动生成 arch_cache.json
```

重启 WebUI，或完全 Reload UI。无需安装额外依赖；优先用 `safetensors.safe_open(...).metadata()` 读取 `__metadata__`，该库不可用或读取失败时回退到标准库直接解析 8 字节长度 + JSON 头。两条路径都不会加载权重。

## 它是怎么工作的

### 1. 架构识别优先级

对每个 Checkpoint，按以下顺序判定，先命中者生效：

1. **手动元数据**：Forge 的 Checkpoint 编辑页写入的同名 `<模型名>.json` 中的 `sd_arch` 字段。
   - 例如 `sd_xl_base_1.0.json` 里的 `"sd_arch": "xl"`。
   - 这一层优先级最高，适合文件名和内置元数据都不可信的模型。
2. **safetensors 头**：优先 `safe_open(...).metadata()`，不可用时直接解析 8 字节头。
   - `modelspec.architecture`（最可靠）
   - `ss_base_model_version`
   - `modelspec.title`、`ss_sd_model_name`（较弱，作为补充）
3. **文件名正则**：`.ckpt`、`.gguf`、无 `__metadata__` 的 safetensors，以及头读取失败时使用。
4. **Civitai 侧车**：文件名仍然为 `unknown` 时，读取 `<模型>.civitai.info` 或 `<模型>.civitai.json` 中的 `baseModel`。
   - 只在前面几层都失败时兜底，不会覆盖已经由文件名正确识别的模型。
   - 只接受 `model.type` 为空或 `Checkpoint` 的侧车；LORA、VAE 等侧车会被忽略。
5. 仍然无法判断：标记为 `unknown`，默认隐藏。

典型元数据：

```text
modelspec.architecture = stable-diffusion-xl-v1-base   -> xl
modelspec.architecture = stable-diffusion-v1           -> sd
modelspec.architecture = stable-diffusion-v3.5-large   -> 不支持（Forge Neo 无 SD3）
ss_base_model_version  = sdxl                          -> xl
```

### Civitai 侧车（`*.civitai.info`）

Civitai Helper 或浏览器下载器通常会在模型旁生成一个 JSON：

```text
model.civitai.info
model.safetensors.civitai.info
model.civitai.json
model.safetensors.civitai.json
```

这四种命名都支持。插件读取其中的 `baseModel`；空文件、`{}`、没有 `baseModel` 的侧车会被忽略，继续用文件名结果。例如：

| Civitai `baseModel` | 识别为 |
| --- | --- |
| `SD 1.4`、`SD 1.5`、`SD 1.5 Hyper`、`SD 1.5 LCM` | `sd` |
| `SDXL 1.0`、`Illustrious`、`Pony`、`NoobAI` | `xl` |
| `Flux.1 D`、`Flux.1 S`、`Flux.1 Krea` | `flux` |
| `Flux.2 Klein` | `klein` |
| `Qwen Image` / `Qwen Image Edit` | `qwen` |
| `Qwen Image 2.1` | `qwen21` |
| `Lumina` / `Lumina 2` | `lumina` |
| `ZImageTurbo` / `ZImageBase` | `zit` |
| `Wan 2.2` / `Wan Video 2.2` | `wan` |
| `Anima` | `anima` |
| `Ernie Image` | `ernie` |
| `PiD` / `PiD 1.5` | `pid` |
| `Krea 2` | `krea` |
| `MOSS` / `MOSS Image` | `moss` |

`SD 2.x`、`SD 3.x`、`Wan 2.1`、`Flux.2 D/Dev` 会标记为 `unsupported`。`Other`、`Boogu` 等 Forge 没有对应 Preset 的值仍保持 `unknown`。

侧车文件可能和模型文件不匹配（例如下载器把别的模型的 JSON 放在旁边），因此它只作为最后一层兜底，并校验 `model.type`。遇到误判时，仍然优先用 **Model Arch** 手动覆盖。

### 2. 缓存

缓存文件：`storage/arch_cache.json`。

- 键：Checkpoint 绝对路径。
- 失效条件：模型 `mtime_ns`、文件大小、同名 `.json` 的 `mtime_ns`、任一 `.civitai.info` / `.civitai.json` 的 `mtime_ns` 发生变化。
- 写入方式：临时文件 + `os.replace`，原子替换。
- 想强制重扫：删除 `storage/arch_cache.json` 后重启/刷新。

### 3. Forge Hook（不修改源码）

Forge 在 `modules_forge/main_entry.py` 里创建 Preset 与 Checkpoint 两个下拉框，并把 Preset 的 `change` 事件绑定到 `main_entry.on_preset_change`。本扩展在 `on_before_ui` 阶段替换两个模块级函数：

- `main_entry.refresh_models`
  - 先调用 Forge 原函数，拿到完整 Checkpoint 列表；
  - 再按 `shared.opts.forge_preset` 过滤；
  - 覆盖启动刷新、刷新按钮、`root_block.load` 三条路径。
- `main_entry.on_preset_change`
  - 先调用 Forge 原函数，保留它返回的 22 个 UI 更新；
  - 只替换第 1 个（Checkpoint 下拉框）：写入过滤后的 `choices`，并把 `value` 修正为列表内仍然存在的模型；
  - 这样不会和 Forge 自己的更新互相覆盖。

`on_script_unloaded` 会还原原函数，避免 Reload UI 后重复包裹。

> 为什么不直接再加一个 `ui_forge_preset.change(... outputs=[ui_checkpoint])`？
> 因为 Forge 的 `on_preset_change` 已经写同一个 `ui_checkpoint` 输出，两个监听器会竞争，最终值取决于事件顺序。包裹原函数是确定性的。

## Preset 对照表

| Preset | 识别的架构 / 常见关键词 |
| --- | --- |
| `sd` | SD1.x；`stable-diffusion-v1`、`sd_v1`、Anything v3/v4/v5、AWPainting v1x |
| `xl` | SDXL；`stable-diffusion-xl`、`sdxl`、Pony、Illustrious、NoobAI、AnythingXL、WAI |
| `flux` | Flux.1；`flux-1`、`flux.1`、Flux Dev/Schnell、Chroma |
| `klein` | Flux.2 / Klein；`flux-2`、`klein` |
| `qwen` | Qwen-Image；`qwen-image` |
| `qwen21` | Qwen-Image-2.1；`qwen-image-2`、`qwen21` |
| `lumina` | Lumina-Image-2.0；`lumina-image-2`、`lumina_2` |
| `zit` | Z-Image-Turbo；`z-image`、`zimage` |
| `wan` | Wan2.2；`wan-2.2`、`wan2.2` |
| `anima` | Anima 2B/2.9B/3.8B；`anima` |
| `ernie` | Ernie-Image；`ernie-image` |
| `pid` | PiD 1.5；`pid` |
| `krea` | Krea2；`krea-2`、`krea2` |
| `moss` | MOSS-Image；`moss-image` |
| `ph3` / `h3i` / `h3` | MiniMax H3 扩展的 Preset；已内置常见文件名规则，可继续补充 |

映射表在 `scripts/preset_model_matcher.py` 的 `PRESET_MAPPING` 与 `_FILENAME_RULES_RAW`。如果安装了新的第三方 Preset：

- Preset 名不在 `PRESET_MAPPING` 中时，扩展**不进行过滤**，显示全部模型，避免把未知 Preset 的下拉框清空。
- 想支持它，在该字典和文件名规则里加一项即可。

### SD2 / SD3

Forge Neo 的 `PresetArch` 只有 SD1、SDXL、Flux.1、Flux.2、Qwen、Lumina、Z-Image、Wan、Anima、Ernie、PiD、Krea2、Qwen-Image-2.1、MOSS-Image。SD2/SD3 已被移除，因此：

- `stable-diffusion-v2*`
- `stable-diffusion-v3*`
- `sd3*`
- `sd2*`

会被标记为 `unsupported`，不会落到 `sd` 下。这是刻意行为，不是漏识别。

### 非图像 / 不兼容模型

有些 ComfyUI 模型目录里混有音频、视频或工具模型。文件名里的 `xl`、`wan` 等字样很容易造成误判，所以下列情况也会被标记为 `unsupported`：

- `acestep*`、`ace-step*`、`minimax_music*`：音频模型；
- `wan2.1*` / `wan-2.1*`：Forge Neo 的 `wan` Preset 是 Wan2.2，不兼容 Wan2.1。

`unsupported` 模型始终隐藏；`unknown` 模型是否显示由 `show_unknown` 决定。

## 设置

WebUI 设置页会出现 **Preset Model Matcher** 分组：

| 设置 | 默认 | 说明 |
| --- | --- | --- |
| `preset_model_matcher_enabled` | 开 | 总开关；关闭后完全恢复 Forge 原生下拉框 |
| `preset_model_matcher_show_unknown` | 关 | 是否显示无法识别架构的模型 |
| `preset_model_matcher_fallback_all` | 开 | 当前 Preset 一个模型都没有时，显示完整列表而不是空列表 |

`fallback_all` 默认开启的原因：Forge 切换 Preset 后会立刻调用 `_load_presets` 加载当前 Checkpoint；如果下拉框为空、值为 `None`，`checkpoint_change` 可能把 `sd_model_checkpoint` 置空，导致后续加载失败。开启后至少保留可选项，并在控制台打印警告。

## 已知限制

- **`.ckpt` / `.gguf` 不解析内部元数据**，只靠文件名；文件名无法识别时，再尝试旁边的 Civitai 侧车。老式 `.ckpt` 是 pickle，安全地读取架构信息需要加载对象，代价和风险都不合适；GGUF 的键值区也没有统一架构字段。
- 文件名正则不可能覆盖所有命名。遇到误判：
  1. 打开 Checkpoint 的元数据编辑页；
  2. 在 **Model Arch** 里选择正确 Preset；
  3. 保存后会生成 `<模型名>.json`，下次分类直接采用，优先级最高。
- `boogu_image_*`、`sam3.1_*`、`sdpose_*` 这类非 Forge 常规 Checkpoint 默认是 `unknown`，会被隐藏。需要显示就把 `show_unknown` 打开。
- 扩展假设 Checkpoint 下拉框仍是 Forge 原生的 `setting_sd_model_checkpoint`。如果其他扩展整体替换了模型选择器，过滤不会生效。
- 备份目录或 ComfyUI 目录里的老模型常常既没有 `modelspec.architecture`，文件名也不含架构标记，且 Civitai 侧车缺失、为空或 `baseModel` 为 `Other`（例如 `Perfect World`、`GuoFeng`、`SilvermoonMix`、`MeinaMIX`、`AbyssOrangeMix`、`Realistic Vision`）。这些仍会被归为 `unknown`。它们不是 Forge 的运行时模型目录，不会影响当前下拉框；若要把它们纳入 Forge，建议用 **Model Arch** 手动标注，而不是往正则里硬编码大量模型系列名。

## 排障

**切换 Preset 后没有变化**

1. 看控制台是否出现：
   ```text
   [Preset Model Matcher] installed checkpoint filter hook
   [Preset Model Matcher] ready: preset='sd', checkpoints=52
   ```
2. 确认 `preset_model_matcher_enabled` 为开。
3. 确认当前 Preset 在 `PRESET_MAPPING` 中；第三方 Preset 默认不过滤。
4. 删除 `storage/arch_cache.json` 再试。

**下拉框变空**

把 `preset_model_matcher_fallback_all` 打开，或给模型设置正确的 **Model Arch**。

**模型被分到错误的 Preset**

优先用 Checkpoint 编辑页的 **Model Arch** 手动覆盖；其次检查 `modelspec.architecture` 是否写了非标准值；最后再调整文件名规则。

**与其他扩展冲突**

本扩展只包裹 `main_entry.refresh_models` 和 `main_entry.on_preset_change`，并在卸载时还原。如果另一个扩展也包裹了这两个函数，最后加载的那个生效；可临时停用另一个扩展确认。

这两个脚本不启动 Forge，也不替代一次真实的 WebUI 重启验证。
