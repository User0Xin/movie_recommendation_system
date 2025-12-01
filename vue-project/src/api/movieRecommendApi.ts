import httpRequest from '@/utils/request'

export const recommendMoviesApi = (uid: string) => {
  return httpRequest({
    url: '/api/recommendMovies',
    method: 'post',
    params: { uid: uid },
  })
}

export const agentQuery = (uid: string,query: string) => {
  return httpRequest({
    url: '/api/agentQuery',
    method: 'post',
    params: { uid: uid, query: query },
  })
}

