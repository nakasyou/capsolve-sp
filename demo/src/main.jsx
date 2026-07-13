import React, { useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import * as ort from 'onnxruntime-web'
import './style.css'

const REPO = 'nakasyou/capsolve-sp'
const MODEL_URL = `https://huggingface.co/${REPO}/resolve/main/model.onnx`
const CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'

// Keep the large ORT WASM runtime out of the app bundle and make the path work
// in both Vite dev server and GitHub Pages builds.
ort.env.wasm.wasmPaths =
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState('')
  const [elapsedMs, setElapsedMs] = useState(null)
  const [status, setStatus] = useState('画像を選択してください')
  const [busy, setBusy] = useState(false)
  const sessionRef = useRef(null)

  const setImageFile = (next) => {
    if (!next) return
    if (!next.type.startsWith('image/')) {
      setStatus('画像ファイルを選択してください')
      return
    }
    setFile(next)
    setResult('')
    setElapsedMs(null)
    setStatus('推論の準備ができました')
    setPreview(URL.createObjectURL(next))
  }

  const selectFile = (event) => setImageFile(event.target.files?.[0])

  const dropFile = (event) => {
    event.preventDefault()
    setImageFile(event.dataTransfer.files?.[0])
  }

  const predict = async () => {
    if (!file) return
    setBusy(true)
    setResult('')
    setStatus('モデルを読み込んでいます…')
    try {
      const loadStarted = performance.now()
      if (!sessionRef.current) {
        sessionRef.current = await ort.InferenceSession.create(MODEL_URL, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        })
      }
      const session = sessionRef.current
      const modelLoadMs = performance.now() - loadStarted
      const bitmap = await createImageBitmap(file)
      if (bitmap.width !== 175 || bitmap.height !== 60) {
        throw new Error('画像サイズは175×60 pxにしてください')
      }
      const canvas = document.createElement('canvas')
      canvas.width = 175
      canvas.height = 60
      const context = canvas.getContext('2d', { willReadFrequently: true })
      context.drawImage(bitmap, 0, 0)
      const pixels = context.getImageData(0, 0, 175, 60).data
      const input = new Float32Array(175 * 60)
      for (let i = 0; i < input.length; i += 1) {
        const gray = 0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2]
        input[i] = (255 - gray) / 255
      }
      const tensor = new ort.Tensor('float32', input, [1, 1, 60, 175])
      const inferenceStarted = performance.now()
      const output = await session.run({ image: tensor })
      const inferenceMs = performance.now() - inferenceStarted
      const logits = output.logits.data
      let text = ''
      for (let position = 0; position < 5; position += 1) {
        let best = 0
        for (let index = 1; index < 36; index += 1) {
          if (logits[position * 36 + index] > logits[position * 36 + best]) best = index
        }
        text += CHARS[best]
      }
      setResult(text)
      setElapsedMs(inferenceMs)
      setStatus(
        modelLoadMs > 1
          ? `モデル準備 ${modelLoadMs.toFixed(0)} ms · CPU推論完了`
          : 'キャッシュ済みモデルでCPU推論完了',
      )
    } catch (error) {
      setStatus(error instanceof Error ? error.message : '推論に失敗しました')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="min-h-screen px-5 py-8 sm:px-10 sm:py-12">
      <div className="mx-auto max-w-5xl">
        <header className="mb-12 flex items-start justify-between gap-6">
          <div>
            <p className="eyebrow">ONNX RUNTIME WEB / CPU</p>
            <h1>
              capsolve<span>-sp</span>
            </h1>
            <p className="mt-4 max-w-xl text-lg leading-8 text-[#625b50]">
              画像はサーバーへ送らず、このブラウザの中だけで5文字を読み取ります。
            </p>
          </div>
          <a
            className="github-link"
            href="https://huggingface.co/nakasyou/capsolve-sp"
            target="_blank"
            rel="noreferrer"
          >
            Model ↗
          </a>
        </header>
        <section className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
          <div className="panel upload-panel">
            <div className="flex items-center justify-between">
              <p className="eyebrow">01 / INPUT</p>
              <span className="chip">175 × 60 px</span>
            </div>
            <label
              className="dropzone"
              htmlFor="captcha-file"
              onDragOver={(event) => event.preventDefault()}
              onDrop={dropFile}
            >
              {preview ? (
                <img src={preview} alt="選択した CAPTCHA" />
              ) : (
                <>
                  <span className="drop-icon">＋</span>
                  <strong>画像をドロップ</strong>
                  <small>またはクリックして選択</small>
                </>
              )}
              <input id="captcha-file" type="file" accept="image/*" onChange={selectFile} />
            </label>
            <div className="mt-6 flex flex-wrap items-center gap-4">
              <button onClick={predict} disabled={!file || busy}>
                {busy ? '推論中…' : '読み取る'}
              </button>
              <span className="status">{status}</span>
            </div>
          </div>
          <div className="panel result-panel">
            <p className="eyebrow">02 / RESULT</p>
            <div className={`result ${result ? 'result-ready' : ''}`}>{result || '— — — — —'}</div>
            <p className="result-note">
              {result
                ? `5文字すべての予測結果 · 推論 ${elapsedMs.toFixed(2)} ms`
                : '結果がここに表示されます'}
            </p>
          </div>
        </section>
        <footer className="mt-8 flex flex-wrap justify-between gap-3 text-sm text-[#847a6d]">
          <span>Model: nakasyou/capsolve-sp · INT8 ONNX</span>
          <a href="https://github.com/nakasyou/capsolve-sp" target="_blank" rel="noreferrer">
            Source on GitHub ↗
          </a>
        </footer>
      </div>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
