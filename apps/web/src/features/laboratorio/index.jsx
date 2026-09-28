import { useState } from 'react'
import { FlaskConical } from 'lucide-react'
import { apiGet } from '../../api'
import DataTable from '../../components/DataTable'

export const meta = {
  id: 'laboratorio',
  title: 'Laboratorio U2-T1',
  unit: 'U2 · Datos',
  icon: FlaskConical,
  order: 2,
}

// Ejercicios de la hoja de trabajo U2-T1 ejecutados contra el API.
const EXERCISES = [
  {
    id: 1,
    name: 'Conteo por región',
    path: '/api/stats/by-region',
    description: 'GROUP BY region',
  },
  {
    id: 2,
    name: 'Perfil de riesgo (score < 650)',
    path: '/api/stats/risk-profile?threshold=650',
    description: 'WHERE credit_score < 650',
  },
  {
    id: 3,
    name: 'Costa por ingreso desc.',
    path: '/api/transactions?region=Costa&order_by_income=true',
    description: 'WHERE region = Costa ORDER BY annual_income DESC',
  },
  {
    id: 4,
    name: 'Tabla completa',
    path: '/api/transactions?limit=5000',
    description: 'SELECT * (equivalente a unir los lotes)',
  },
]

export default function Laboratorio() {
  const [result, setResult] = useState(null)
  const [active, setActive] = useState(null)
  const [error, setError] = useState(null)

  const run = async (exercise) => {
    setActive(exercise.id)
    setError(null)
    try {
      setResult({ title: exercise.name, rows: await apiGet(exercise.path) })
    } catch (e) {
      setResult(null)
      setError(e.message)
    }
  }

  return (
    <div className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
      <h3 className="text-sm font-semibold text-slate-700">
        Ejercicios de extracción — hoja de trabajo U2-T1
      </h3>
      <div className="mt-3 flex flex-wrap gap-2">
        {EXERCISES.map((ex) => (
          <button
            key={ex.id}
            onClick={() => run(ex)}
            title={ex.description}
            className={`rounded-lg px-3 py-2 text-sm font-medium transition ${active === ex.id
                ? 'bg-indigo-600 text-white'
                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
              }`}
          >
            {ex.id}. {ex.name}
          </button>
        ))}
      </div>
      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      <div className="mt-4">
        {result && <DataTable title={result.title} rows={result.rows} />}
      </div>
    </div>
  )
}
