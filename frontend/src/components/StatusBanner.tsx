export function StatusBanner({ notes, error }: { notes: string[]; error: string | null }) {
  if (!notes.length && !error) return null;

  return (
    <div className="space-y-2">
      {error && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-2 text-sm text-rose-300">
          {error}
        </div>
      )}
      {notes.map((note, i) => (
        <div
          key={i}
          className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2 text-sm text-amber-300"
        >
          {note}
        </div>
      ))}
    </div>
  );
}
