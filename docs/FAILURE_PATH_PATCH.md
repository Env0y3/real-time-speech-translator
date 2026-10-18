# 107b76d 后最小故障路径补丁

## 版本与复现

父提交：107b76d204d261c40f20254d9638aa89726ff229。原工作目录的 f216420 源码树与该远端提交完全一致；保留旧目录、未提交成果及原证据，在独立 worktree 从远端父提交增量承接。没有重写历史、删除文件或合并 main。远端分支仍为 fix/desktop-reproducibility-stability。

受测 core/gui/tests 文件 SHA-256 清单见 evidence/patch-source-manifest.json，聚合标识为 7b9d9edb83138fa25c52e14e9e8b15b7c10dc8224afb21fc58938a71e2c359a5。命令、环境和退出码见 evidence/patch-validation.json。这里的结果仅属于该补丁，不沿用原81项成绩。

## 复现、原因、修复与验证

1. 同名设备：原历史恢复规则也用于用户刚选择的设备，重复名称会在开始前刷新时丢失。现在仅在完整枚举快照保持相同、当前选择有效时保留明确选择；历史同名歧义仍拒绝。快照变化、枚举失败、旧索引复用或同名设备减少时要求重选，且不会下一次刷新就自动选择剩余同名设备。提示说明无法确认身份及重新选择操作。新增10项离线Qt设备测试，修复前4失败/2错误/4通过，修复后10通过。
2. 合成故障：原路径只有 partial_failure 标记，可能最终回到 Listening。现在区分 tts_service_failed、tts_partial_failed、audio_output_failed；失败事件立即请求停止并保留界面错误。零音频且允许默认输出降级时，记录 tts_fallback started/succeeded/failed 及日志中的 service_failure，成功降级不混成主服务成功；已收部分音频不做整句降级重播。新增9项实际异步TTS编排/播放消费者故障测试，WebSocket、输出和本地合成使用离线替身。
3. 输出写入：在实际播放 worker 的 write 异常处立即发事件并置停止信号，取消等待远端句尾或下一翻译段；不依赖句尾日志才通知。PipelineThread在Qt排队前同步请求停止。清理及迟到状态不能覆盖首错。
4. 复查额外发现：最终线程异常会把已分类界面错误覆盖成 pipeline_error。真实TTS故障→Qt线程→界面整链测试先复现三种错误均被覆盖，修复后通过；迟到另一错误及Listening也不覆盖首错。相关3个测试方法覆盖这些流程，并经独立复查重跑通过。

前置失败证据见 evidence/patch-device-before.txt、patch-tts-before.txt、patch-gui-before.txt 和 patch-gui-signal-before.txt。前置阶段测试集曾增长，不把不同阶段数量作最终合计。

## 最终完整回归

复用现有 Debian Linux x86_64、CPython 3.12.14 的两个隔离环境；不联网调用服务，不访问真实声卡。Qt使用offscreen。

- 核心+GUI：103计划/103执行/103通过/0失败/0跳过/0模块导入错误/0其他错误/0未运行，退出0，完整正常结束。
- 无GUI依赖：103计划/103执行/66通过/37明确跳过（PySide6未安装）/0失败/0模块导入错误/0其他错误/0未运行，退出0，完整正常结束。
- git diff --check：退出0。

最终原始逐项输出与分类JSON见 evidence/patch-full-*.txt 和 patch-full-*-results.json。最初恢复环境时曾误配PortAudio动态库目录，造成9模块导入错误；纠正到 root/usr/lib/x86_64-linux-gnu 后完整重跑以上结果，未将此环境错误当作产品缺陷或缺GUI跳过。

## 真实设备验收边界

本轮没有重新进行模型下载/真实模型性能测试，也没有真实麦克风、普通扬声器、虚拟声卡、监听、付费合成或翻译服务验收。Windows真机均未验收。原模型证据保留，但不计为本补丁的新成绩。

PortAudio枚举元数据不是硬件唯一ID：设备替换但所有暴露字段完全相同时无法证明身份，因此运行中的热插拔最终仍需真实设备检查。测试验证错误通知及停止请求及时发生；真实WebSocket关闭仍受现有10秒close_timeout及系统调度影响，不宣称硬件/网络固定退出时限。

下一步按 docs/DESKTOP_ACCEPTANCE.md 准备Windows真实设备。补查：同名设备明确重选→开始前刷新；拔掉所选输出→不得转向其它设备；零音频/部分音频服务中断→错误分类/停止/不重复整句；输出故障发生后→立即错误、随后清理完成仍保留错误。记录提交、系统/设备、输入样本、界面截图及脱敏日志。仅在已有本项目凭据和预算下验证服务，不新增费用。
