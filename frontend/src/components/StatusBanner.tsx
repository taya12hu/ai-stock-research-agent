export function StatusBanner({ notes, error }: { notes: string[]; error: string | null }) {
  if (!notes.length && !error) return null;

  return (
    <div className="space-y-2">
      {error && (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-2 text-sm text-rose-700">
          {error}
        </div>
      )}
      {notes.map((note, i) => (
        <div
          key={i}
          className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800"
        >
          {note}
        </div>
      ))}
    </div>
  );
}
