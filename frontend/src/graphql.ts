import { ApolloClient, HttpLink, InMemoryCache, gql, type ApolloError } from '@apollo/client';
import { UNREACHABLE, type ModelInfo } from './api';

const BASE: string = import.meta.env?.VITE_API_BASE ?? '/api';
const GATEWAY = new Set([502, 503, 504]);

/** One client per app (and per test), so the cache never leaks between them. */
export function makeClient() {
  return new ApolloClient({
    // fetch is looked up on every request, so tests can stub it
    link: new HttpLink({ uri: `${BASE}/graphql`, fetch: (input, init) => fetch(input, init) }),
    cache: new InMemoryCache(),
  });
}

export const MODEL_INFO_QUERY = gql`
  query ModelInfo {
    modelInfo {
      model
      version
      datasetSize
      majorityBaseline
      features
      testMetrics {
        accuracy
        f1Macro
        logLoss
      }
    }
  }
`;

export interface ModelInfoQuery {
  modelInfo: {
    model: string;
    version: string;
    datasetSize: number;
    majorityBaseline: number | null;
    features: string[];
    testMetrics: { accuracy: number | null; f1Macro: number | null; logLoss: number | null };
  };
}

/** The GraphQL shape, mapped onto the ModelInfo type the components already use. */
export function toModelInfo(m: ModelInfoQuery['modelInfo']): ModelInfo {
  return {
    model: m.model,
    version: m.version,
    dataset_size: m.datasetSize,
    majority_baseline: m.majorityBaseline,
    features: m.features,
    test_metrics: {
      accuracy: m.testMetrics.accuracy ?? undefined,
      f1_macro: m.testMetrics.f1Macro ?? undefined,
      log_loss: m.testMetrics.logLoss ?? undefined,
    },
  };
}

/** Same user facing messages as the REST client: gateway errors and network failures read as "service unavailable". */
export function describeError(err: ApolloError): string {
  const net = err.networkError as (Error & { statusCode?: number }) | null;
  if (net) {
    if (!net.statusCode || GATEWAY.has(net.statusCode)) return UNREACHABLE;
    return `Request failed (${net.statusCode})`;
  }
  return err.graphQLErrors[0]?.message ?? err.message;
}
