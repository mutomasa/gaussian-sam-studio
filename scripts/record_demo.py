"""bonsai の全体の流れ（3DGS → 描画 → SAM 3.1 → 3D セグメンテーション → 評価）をデモ動画に録る。

    uv run python scripts/record_demo.py [--out outputs/demo/bonsai_demo.mp4] [--keep-app]

- Streamlit をこのスクリプトが起動・終了する（ポート 8599）。ブラウザはシステムの Google Chrome
- 学習済みの 3DGS（outputs/training/ の完了ジョブ）と、実画像 vs 3D 描画の比較結果が既にある前提
- GPU の待ち時間（描画・SAM・持ち上げ）は録画後に早送りにする（字幕に「⏩ 早送り」を出す）
- 進め方は UI Demo Video Recorder の考え方に合わせた: 下見 → リハーサル → 録画、カーソル表示、
  人が打つ速さの入力、操作ごとの間
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright

from spatialsam.config import PROJECT_ROOT

URL = "http://localhost:8599"
VIEWPORT = {"width": 1440, "height": 900}
FAST_FORWARD = 8.0  # 待ち時間の早送り倍率

CURSOR_JS = """() => {
  if (document.getElementById('demo-cursor')) return;
  const c = document.createElement('div');
  c.id = 'demo-cursor';
  c.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M5 3L19 12L12 13L9 20L5 3Z" fill="white" stroke="black" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  c.style.cssText = 'position:fixed;z-index:999999;pointer-events:none;left:0;top:0;transition:left .08s,top .08s;filter:drop-shadow(1px 1px 2px rgba(0,0,0,.35))';
  document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; });
  const s = document.createElement('div');
  s.id = 'demo-subtitle';
  s.style.cssText = 'position:fixed;z-index:999998;pointer-events:none;left:50%;bottom:28px;transform:translateX(-50%);max-width:80%;padding:10px 22px;border-radius:10px;background:rgba(15,15,20,.82);color:#fff;font:600 22px/1.4 "Noto Sans CJK JP","Hiragino Sans",sans-serif;text-align:center;opacity:0;transition:opacity .3s';
  document.body.appendChild(s);
}"""

SUBTITLE_JS = """(text) => {
  const s = document.getElementById('demo-subtitle');
  if (!s) return;
  s.textContent = text || '';
  s.style.opacity = text ? '1' : '0';
}"""


@dataclass
class Recorder:
    """操作の補助と、早送りにする区間（録画開始からの秒数）の記録。"""

    page: Page
    started: float = field(default_factory=time.monotonic)
    fast_segments: list[tuple[float, float]] = field(default_factory=list)
    subtitle: str = ""

    def now(self) -> float:
        return time.monotonic() - self.started

    def prepare_page(self) -> None:
        self.page.evaluate(CURSOR_JS)
        self.page.evaluate(SUBTITLE_JS, self.subtitle)

    def say(self, text: str, hold: float = 2.0) -> None:
        self.subtitle = text
        self.prepare_page()
        self.pause(hold)

    def pause(self, seconds: float) -> None:
        self.page.wait_for_timeout(seconds * 1000)

    def move_to(self, target: Locator) -> None:
        target.scroll_into_view_if_needed()
        box = target.bounding_box()
        if box:
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=18)
        self.pause(0.4)

    def click(self, target: Locator, after: float = 1.5) -> None:
        self.prepare_page()
        self.move_to(target)
        target.click()
        self.pause(after)

    def type(self, target: Locator, text: str) -> None:
        self.click(target, after=0.3)
        target.fill("")
        target.press_sequentially(text, delay=70)
        target.press("Enter")
        self.pause(1.0)

    def wait_fast(self, condition, label: str, timeout: float = 600) -> None:
        """GPU の処理待ち。録画後にこの区間を早送りにする。"""
        previous = self.subtitle
        self.say(f"{label}　⏩ 早送り", hold=0.5)
        start = self.now()
        condition(timeout * 1000)
        self.fast_segments.append((start, self.now()))
        self.say(previous, hold=0.3)


def wait_text(page: Page, text: str, state: str = "visible"):
    return lambda timeout: page.get_by_text(text).first.wait_for(state=state, timeout=timeout)


def open_page(rec: Recorder, name: str, label: str) -> None:
    rec.click(rec.page.locator(f'a[href$="{name}"]'), after=0.5)
    rec.page.get_by_role("heading", name=label).wait_for(timeout=60_000)
    rec.page.wait_for_load_state("networkidle")
    rec.pause(1.5)
    rec.prepare_page()


def select_option(rec: Recorder, combobox_label: str, option_text: str) -> None:
    rec.click(rec.page.get_by_role("combobox", name=combobox_label), after=0.6)
    rec.click(rec.page.get_by_role("option").filter(has_text=option_text).first, after=1.2)


def set_number(rec: Recorder, label: str, value: int) -> None:
    rec.type(rec.page.get_by_role("spinbutton", name=label), str(value))


def scene_step(rec: Recorder) -> None:
    page = rec.page
    open_page(rec, "3d_rendering", "3D Scene & Rendering")
    rec.say("Mip-NeRF 360 bonsai を gsplat で学習した 3DGS（Gaussian 100 万個、評価用視点で PSNR 30.7）")
    rec.click(page.get_by_role("tab", name="3DGS 学習"))
    rec.pause(2.5)
    rec.say("学習済みのシーンを描画タブに渡す")
    rec.click(page.get_by_role("button", name="この 3DGS を描画タブの入力に設定"))
    rec.click(page.get_by_role("tab", name="描画"))
    page.get_by_text("カメラ軌道").first.wait_for(timeout=120_000)
    rec.say("撮影カメラの軌道で描画（学習時の評価用視点は除く）")
    rec.click(page.get_by_text("データセットの撮影カメラ（COLMAP）"))
    set_number(rec, "間引き（stride）", 3)
    set_number(rec, "長辺 [px]", 780)
    set_number(rec, "評価用視点を除く間隔（0 = 除かない）", 8)
    rec.click(page.get_by_role("button", name="描画", exact=True), after=0.5)
    rec.wait_fast(wait_text(page, "完了"), "85 視点の RGB・深度・カメラ姿勢を描画中")
    rec.say("RGB と深度の動画ができた", hold=0.5)
    page.get_by_text("RGB（").first.scroll_into_view_if_needed()
    rec.pause(4.0)
    rec.click(page.get_by_role("button", name="この動画を Video Segmentation で使う"), after=0.5)


def segmentation_step(rec: Recorder) -> None:
    page = rec.page
    page.get_by_role("heading", name="Video Segmentation").wait_for(timeout=60_000)
    rec.pause(1.5)
    rec.say("描画した動画に、テキストで「bonsai」と指示して SAM 3.1 で推論")
    rec.type(page.get_by_role("textbox", name="テキストプロンプト"), "bonsai")
    rec.click(page.get_by_role("button", name="推論を実行"), after=0.5)
    rec.wait_fast(wait_text(page, "マスク重畳"), "SAM 3.1（オープンウェイト）をローカル GPU で推論中")
    rec.say("盆栽だけがフレームを通して追跡されている", hold=0.5)
    page.get_by_text("マスク重畳").first.scroll_into_view_if_needed()
    rec.pause(5.0)
    slider = page.get_by_role("slider", name="フレーム")
    rec.move_to(slider)
    slider.focus()
    for _ in range(12):
        page.keyboard.press("ArrowRight")
        rec.pause(0.25)
    rec.pause(2.0)


def lifting_step(rec: Recorder) -> None:
    page = rec.page
    open_page(rec, "3d_segmentation", "3D Segmentation")
    rec.say("2D マスクを、描画時の深度とカメラ姿勢で 3D の Gaussian に持ち上げる")
    rec.click(page.get_by_role("button", name="3D に持ち上げる"), after=0.5)
    rec.wait_fast(wait_text(page, "各視点の投影と投票をしています", state="hidden"),
                  "85 視点で可視性つきの投票")
    page.get_by_text("元のシーン").first.wait_for(timeout=120_000)
    rec.say("19 万個の Gaussian が「bonsai」に。対象を強調・抽出できる", hold=0.5)
    page.get_by_text("元のシーン").first.scroll_into_view_if_needed()
    rec.pause(3.0)
    slider = page.get_by_role("slider", name="視点（描画に使ったカメラ）")
    rec.move_to(slider)
    slider.focus()
    for _ in range(6):
        for _ in range(5):
            page.keyboard.press("ArrowRight")
        rec.pause(1.2)
    rec.say("学習にも持ち上げにも使っていない視点の実画像で評価")
    rec.click(page.get_by_role("button", name="評価する"), after=0.5)
    rec.wait_fast(wait_text(page, "再投影 IoU（平均）"), "評価用視点にラベルを描画")
    page.get_by_text("再投影 IoU（平均）").first.scroll_into_view_if_needed()
    rec.say("新しい視点でも実画像の SAM と IoU 0.9 前後で一致", hold=5.0)


def evaluation_step(rec: Recorder) -> None:
    page = rec.page
    open_page(rec, "evaluation", "Evaluation & Experiments")
    rec.click(page.get_by_role("tab", name="実画像 vs 3D 描画"))
    page.get_by_text("評価用視点").first.wait_for(timeout=60_000)
    rec.say("実画像・学習した 3DGS・点群の描画を、画質と SAM の結果で比較")
    page.get_by_text("画質（実画像との比較）").first.scroll_into_view_if_needed()
    rec.pause(4.0)
    slider = page.get_by_role("slider", name="視点")
    rec.move_to(slider)
    slider.focus()
    for _ in range(4):
        for _ in range(4):
            page.keyboard.press("ArrowRight")
        rec.pause(1.5)
    rec.say("学習した 3DGS なら SAM の結果は実画像とほぼ同じ。点群の描画では盆栽を検出できない", hold=5.0)
    rec.say("SpatialSAM Studio — オープンウェイト・ローカル GPU だけで、テキストから動画と 3D をセグメント", hold=4.0)


def run_demo(page: Page) -> Recorder:
    rec = Recorder(page)
    page.goto(URL)
    page.get_by_role("heading", name="SpatialSAM Studio").wait_for(timeout=120_000)
    rec.say("SpatialSAM Studio：テキスト指定の動画セグメンテーションを 3D Gaussian Splatting とつなぐ", hold=3.5)
    scene_step(rec)
    segmentation_step(rec)
    lifting_step(rec)
    evaluation_step(rec)
    return rec


def start_app() -> subprocess.Popen:
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                                "--server.port", "8599"], cwd=PROJECT_ROOT,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"{URL}/_stcore/health", timeout=1)
            return process
        except OSError:
            time.sleep(1)
    process.terminate()
    raise RuntimeError("Streamlit が起動しません")


def speed_up(raw: Path, out: Path, segments: list[tuple[float, float]], offset: float) -> None:
    """録画（WebM）を MP4 にし、待ち区間だけ FAST_FORWARD 倍速にする。offset は録画開始から計測開始までの秒数。"""
    import imageio_ffmpeg

    cuts, cursor = [], 0.0
    for start, end in segments:
        start, end = start + offset, end + offset
        cuts += [(cursor, start, 1.0), (start, end, FAST_FORWARD)]
        cursor = end
    cuts.append((cursor, None, 1.0))
    parts, labels = [], []
    for i, (a, b, speed) in enumerate(c for c in cuts if c[1] is None or c[1] - c[0] > 0.05):
        trim = f"trim=start={a:.3f}" + (f":end={b:.3f}" if b is not None else "")
        parts.append(f"[0:v]{trim},setpts=(PTS-STARTPTS)/{speed}[v{i}]")
        labels.append(f"[v{i}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=1:a=0[out]"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw),
                    "-filter_complex", graph, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-r", "30", str(out)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "demo" / "bonsai_demo.mp4")
    parser.add_argument("--keep-app", action="store_true", help="既に起動している Streamlit を使う")
    args = parser.parse_args()
    raw_dir = args.out.parent / "raw"
    app = None if args.keep_app else start_app()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
            context = browser.new_context(viewport=VIEWPORT, record_video_dir=str(raw_dir),
                                          record_video_size=VIEWPORT, locale="ja-JP")
            page = context.new_page()
            launched = time.monotonic()
            rec = run_demo(page)
            offset = rec.started - launched  # 録画はページを作った時点から始まる
            video = page.video
            context.close()
            browser.close()
            raw = Path(video.path())
    finally:
        if app is not None:
            app.terminate()
    speed_up(raw, args.out, rec.fast_segments, offset)
    (args.out.with_suffix(".json")).write_text(json.dumps({"fast_segments": rec.fast_segments}, indent=2))
    print(f"→ {args.out}（元の録画: {raw}）")


if __name__ == "__main__":
    main()
