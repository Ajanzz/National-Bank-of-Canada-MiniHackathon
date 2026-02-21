import React from "react";
import * as Popover from "@radix-ui/react-popover";
import * as Select from "@radix-ui/react-select";
import { DayPicker } from "react-day-picker";
import { format as formatDate } from "date-fns";
import "react-day-picker/dist/style.css";

export type PnlCalendarMode = "month" | "year";

export type PnlCalendarValue = {
  year: number;
  month?: number; // 0-11
};

type PnlCalendarControlProps = {
  mode: PnlCalendarMode;
  value: PnlCalendarValue;
  onChange: (next: PnlCalendarValue) => void;
  minYear?: number;
  maxYear?: number;
};

const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] as const;

function clampMonth(input: number | undefined) {
  if (!Number.isFinite(input)) return 0;
  const month = Math.floor(input as number);
  if (month < 0) return 0;
  if (month > 11) return 11;
  return month;
}

function buildYearOptions(minYear: number, maxYear: number) {
  const years: number[] = [];
  for (let year = maxYear; year >= minYear; year -= 1) years.push(year);
  return years;
}

export function PnlCalendarControl({
  mode,
  value,
  onChange,
  minYear,
  maxYear,
}: PnlCalendarControlProps) {
  const today = React.useMemo(() => new Date(), []);
  const nowYear = today.getFullYear();
  const lowerYear = minYear ?? nowYear - 15;
  const upperYear = maxYear ?? nowYear + 5;
  const selectedMonth = clampMonth(value.month ?? today.getMonth());
  const selectedDate = new Date(value.year, selectedMonth, 1);
  const [displayMonth, setDisplayMonth] = React.useState<Date>(selectedDate);

  React.useEffect(() => {
    setDisplayMonth(selectedDate);
  }, [selectedDate.getFullYear(), selectedDate.getMonth()]);

  const yearOptions = React.useMemo(() => buildYearOptions(lowerYear, upperYear), [lowerYear, upperYear]);
  const pillLabel = mode === "month" ? formatDate(selectedDate, "MMM yyyy") : `Year ${value.year}`;

  const moveMonth = (direction: -1 | 1) => {
    if (mode === "year") {
      onChange({ year: value.year + direction, month: selectedMonth });
      return;
    }

    const next = new Date(value.year, selectedMonth + direction, 1);
    onChange({ year: next.getFullYear(), month: next.getMonth() });
    setDisplayMonth(next);
  };

  const setToday = () => {
    onChange({ year: nowYear, month: today.getMonth() });
    setDisplayMonth(new Date(nowYear, today.getMonth(), 1));
  };

  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" className="pnlCalTrigger">
          {pillLabel}
        </button>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content className="pnlCalPopover" align="start" sideOffset={10}>
          <div className="pnlCalActions">
            <button type="button" className="pnlCalNavBtn" onClick={() => moveMonth(-1)}>
              Prev
            </button>
            <button type="button" className="pnlCalNavBtn" onClick={() => moveMonth(1)}>
              Next
            </button>
            <button type="button" className="pnlCalTodayBtn" onClick={setToday}>
              Today
            </button>
          </div>

          {mode === "month" ? (
            <DayPicker
              mode="single"
              month={displayMonth}
              selected={selectedDate}
              onMonthChange={setDisplayMonth}
              onSelect={(picked) => {
                if (!picked) return;
                onChange({ year: picked.getFullYear(), month: picked.getMonth() });
                setDisplayMonth(new Date(picked.getFullYear(), picked.getMonth(), 1));
              }}
              className="pnlCalDayPicker"
              captionLayout="dropdown-buttons"
              fromYear={lowerYear}
              toYear={upperYear}
            />
          ) : (
            <div className="pnlCalYearSelect">
              <label className="pnlCalLabel">Year</label>
              <Select.Root
                value={String(value.year)}
                onValueChange={(nextValue) => {
                  const nextYear = Number(nextValue);
                  if (!Number.isFinite(nextYear)) return;
                  onChange({ year: nextYear, month: selectedMonth });
                }}
              >
                <Select.Trigger className="pnlCalSelectTrigger">
                  <Select.Value placeholder="Select year" />
                </Select.Trigger>
                <Select.Portal>
                  <Select.Content className="pnlCalSelectContent" position="popper" sideOffset={6}>
                    <Select.Viewport>
                      {yearOptions.map((year) => (
                        <Select.Item key={year} value={String(year)} className="pnlCalSelectItem">
                          <Select.ItemText>{year}</Select.ItemText>
                        </Select.Item>
                      ))}
                    </Select.Viewport>
                  </Select.Content>
                </Select.Portal>
              </Select.Root>
              <div className="pnlCalHint">
                Aggregates monthly PnL for {value.year}.
              </div>
            </div>
          )}

          <Popover.Arrow className="pnlCalArrow" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
