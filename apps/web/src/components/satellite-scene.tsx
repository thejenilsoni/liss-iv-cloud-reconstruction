type SatelliteSceneProps = {
  variant: "cloudy" | "clear" | "mask" | "uncertainty";
  className?: string;
};

export function SatelliteScene({
  variant,
  className,
}: SatelliteSceneProps) {
  const showClouds = variant === "cloudy";
  const isMask = variant === "mask";
  const isUncertainty = variant === "uncertainty";

  return (
    <svg
      className={className}
      viewBox="0 0 840 520"
      role="img"
      aria-label={`${variant} satellite scene preview`}
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <linearGradient id={`soil-${variant}`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor={isMask ? "#11151a" : "#71825a"} />
          <stop offset=".45" stopColor={isMask ? "#0c1115" : "#445c3e"} />
          <stop offset="1" stopColor={isMask ? "#12171a" : "#9a7c4f"} />
        </linearGradient>
        <linearGradient id={`river-${variant}`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor={isMask ? "#151a20" : "#6aa4a3"} />
          <stop offset="1" stopColor={isMask ? "#0c1115" : "#244f57"} />
        </linearGradient>
        <radialGradient id={`heat-${variant}`}>
          <stop stopColor="#fb665d" stopOpacity=".9" />
          <stop offset=".5" stopColor="#f5c266" stopOpacity=".45" />
          <stop offset="1" stopColor="#5be2bf" stopOpacity="0" />
        </radialGradient>
        <filter id={`soft-${variant}`}>
          <feGaussianBlur stdDeviation="10" />
        </filter>
        <filter id={`cloud-${variant}`}>
          <feGaussianBlur stdDeviation="7" />
        </filter>
        <pattern
          id={`grid-${variant}`}
          width="58"
          height="58"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(18)"
        >
          <path
            d="M0 0v58M0 0h58"
            stroke={isMask ? "#33404a" : "#d5d7a7"}
            strokeOpacity=".23"
            strokeWidth="2"
          />
        </pattern>
      </defs>

      <rect width="840" height="520" fill={`url(#soil-${variant})`} />
      <rect width="840" height="520" fill={`url(#grid-${variant})`} />

      <g opacity={isMask ? 0.4 : 0.88}>
        <path
          d="M-50 78C80 10 169 17 266 77s197 24 274-10 187-40 350 45v96c-156-68-253-56-351-4s-200 69-309 4S69 153-50 197Z"
          fill="#263c31"
        />
        <path
          d="M-30 405c132-91 244-78 327-9s165 67 256 20 187-42 322 47v77H-30Z"
          fill="#293f33"
        />
        <path
          d="M-20 286c153-47 247-8 354 39s211 25 306-44 180-83 250-65v88c-87-9-163 15-243 75s-209 87-330 27S121 338-20 379Z"
          fill={`url(#river-${variant})`}
        />
        <path
          d="m34 27 219 491M176-23l214 543M564-18l-54 552M742-18l-91 552"
          stroke={isMask ? "#30383f" : "#d3cba0"}
          strokeOpacity=".36"
          strokeWidth="8"
        />
        <path
          d="M-30 147 878 411M-30 49l899 269M-40 323l905 139"
          stroke={isMask ? "#30383f" : "#c4be96"}
          strokeOpacity=".3"
          strokeWidth="6"
        />
      </g>

      {isUncertainty && (
        <g>
          <circle cx="350" cy="195" r="190" fill={`url(#heat-${variant})`} />
          <circle cx="600" cy="305" r="155" fill={`url(#heat-${variant})`} />
          <circle cx="215" cy="400" r="120" fill={`url(#heat-${variant})`} />
        </g>
      )}

      {(showClouds || isMask) && (
        <g
          filter={`url(#cloud-${variant})`}
          fill={isMask ? "#f4f7f6" : "#fff"}
          opacity={isMask ? 0.98 : 0.93}
        >
          <ellipse cx="357" cy="166" rx="151" ry="81" />
          <ellipse cx="472" cy="220" rx="168" ry="98" />
          <ellipse cx="590" cy="289" rx="143" ry="86" />
          <ellipse cx="247" cy="278" rx="111" ry="63" />
          <ellipse cx="671" cy="187" rx="91" ry="55" />
        </g>
      )}

      {showClouds && (
        <g
          filter={`url(#soft-${variant})`}
          fill="#17201e"
          opacity=".38"
          transform="translate(27 32)"
        >
          <ellipse cx="357" cy="166" rx="151" ry="81" />
          <ellipse cx="472" cy="220" rx="168" ry="98" />
          <ellipse cx="590" cy="289" rx="143" ry="86" />
        </g>
      )}

      <path
        d="M25 25h55M25 25v55M815 25h-55M815 25v55M25 495h55M25 495v-55M815 495h-55M815 495v-55"
        fill="none"
        stroke="#eaf3ef"
        strokeOpacity=".58"
        strokeWidth="2"
      />
    </svg>
  );
}

